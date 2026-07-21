"""FastAPI service for the malaria cell classification pipeline.

Endpoints
    GET  /                  service metadata
    GET  /health            liveness probe (cheap, no model touch)
    GET  /status            uptime, model version, last training event
    GET  /metrics           evaluation metrics of the live model
    GET  /visualizations    derived feature distributions for the dashboard
    POST /predict           classify one uploaded cell image
    POST /predict/batch     classify several uploaded cell images
    POST /upload            stage bulk labelled data for retraining
    GET  /uploads           how much staged data is waiting
    POST /retrain           trigger retraining on the staged data
    GET  /retrain/status    progress of the running or last retrain job

Retraining runs in a background thread rather than inside the request. A
retrain takes minutes; holding the HTTP connection open for it would time out
behind most cloud load balancers and would block the worker from serving
predictions the whole time.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MODEL_PATH, read_metadata  # noqa: E402
from src.preprocessing import (  # noqa: E402
    CLASS_NAMES,
    UPLOAD_DIR,
    count_uploads,
    save_uploaded_archive,
    save_uploaded_images,
)
from src.prediction import (  # noqa: E402
    get_model,
    is_model_available,
    predict,
    predict_batch,
)

SERVICE_STARTED_AT = time.time()

# Retraining on too few images overfits the model to that handful and can
# undo a good baseline, so the endpoint refuses below this many staged images.
MIN_IMAGES_TO_RETRAIN = 20

# How many original training images per class to replay alongside the new
# uploads. Enough to anchor the baseline distribution without making a
# retrain as expensive as training from scratch.
BASE_REPLAY_PER_CLASS = 750
RETRAIN_SEED = 42

# Labelled example cells bundled into the image so the hosted dashboard is
# usable without the dataset.
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

app = FastAPI(
    title="Malaria Cell Classification API",
    description=(
        "End-to-end ML pipeline for detecting malaria parasites in segmented "
        "red blood cell images."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Retrain job state
# ---------------------------------------------------------------------------

class RetrainJob:
    """Tracks the single in-flight retrain job.

    A lock guards against two clients hitting /retrain at once, which would
    otherwise have two threads writing the same model file concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "idle"
        self.message = "No retraining has been run yet."
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.metrics: dict | None = None

    def try_claim(self) -> bool:
        with self._lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.message = "Retraining in progress."
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.finished_at = None
            self.metrics = None
            return True

    def finish(self, status: str, message: str, metrics: dict | None = None) -> None:
        with self._lock:
            self.status = status
            self.message = message
            self.metrics = metrics
            self.finished_at = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": self.metrics,
        }


retrain_job = RetrainJob()


class RetrainRequest(BaseModel):
    epochs: int = 5
    learning_rate: float = 1e-4
    include_base_data: bool = True


# ---------------------------------------------------------------------------
# Informational endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "service": "Malaria Cell Classification API",
        "version": "1.0.0",
        "classes": list(CLASS_NAMES),
        "docs": "/docs",
        "model_available": is_model_available(),
    }


@app.get("/health")
def health() -> dict:
    """Cheap liveness probe. Deliberately does not touch the model."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/status")
def status() -> dict:
    """Uptime and live-model information for the dashboard."""
    uptime_seconds = time.time() - SERVICE_STARTED_AT
    metadata = read_metadata()

    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "status": "online",
        "uptime_seconds": round(uptime_seconds, 1),
        "uptime_human": f"{hours}h {minutes}m {seconds}s",
        "started_at": datetime.fromtimestamp(
            SERVICE_STARTED_AT, tz=timezone.utc
        ).isoformat(),
        "model_available": is_model_available(),
        "model_version": metadata.get("version", 0),
        "last_training_event": metadata.get("latest"),
        "retrain": retrain_job.snapshot(),
        "staged_uploads": count_uploads(),
    }


@app.get("/samples")
def samples(limit_per_class: int = 3) -> dict:
    """List the bundled example cells.

    The deployed container carries no dataset, so a user opening the hosted
    dashboard has no images to try. Labelled examples ship with the image so
    both prediction and retraining are demonstrable without cloning the
    repository or downloading 353 MB from NIH.

    `limit_per_class` caps how many are listed — the prediction page wants a
    handful of thumbnails, not all 50.
    """
    if not SAMPLES_DIR.is_dir():
        return {"samples": [], "count": 0, "available": 0}

    entries, available = [], 0
    for class_name in CLASS_NAMES:
        class_dir = SAMPLES_DIR / class_name
        if not class_dir.is_dir():
            continue
        paths = sorted(class_dir.glob("*.png"))
        available += len(paths)
        for path in paths[: max(1, limit_per_class)]:
            entries.append(
                {
                    "label": class_name,
                    "filename": path.name,
                    "url": f"/samples/{class_name}/{path.name}",
                }
            )

    return {"samples": entries, "count": len(entries), "available": available}


@app.get("/samples/archive")
def samples_archive(per_class: int = 25):
    """Build a labelled ZIP of bundled cells, ready to upload for retraining.

    This is what makes the retraining flow demonstrable on the hosted app.
    Generating the archive from `data/test/` on the local filesystem — as an
    earlier version of the dashboard did — cannot work in the deployed
    container, which ships no dataset.

    The archive uses the `Parasitized/` + `Uninfected/` folder layout that
    POST /upload expects, so it round-trips through the pipeline unmodified.
    """
    import io
    import zipfile

    from fastapi.responses import StreamingResponse

    if not SAMPLES_DIR.is_dir():
        raise HTTPException(status_code=404, detail="No bundled samples available.")

    per_class = max(1, min(per_class, 100))
    buffer = io.BytesIO()
    written = 0

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for class_name in CLASS_NAMES:
            class_dir = SAMPLES_DIR / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.glob("*.png"))[:per_class]:
                archive.write(path, f"{class_name}/{path.name}")
                written += 1

    if written == 0:
        raise HTTPException(status_code=404, detail="No sample images found.")

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="retrain_batch.zip"',
            "X-Image-Count": str(written),
        },
    )


@app.get("/samples/{label}/{filename}")
def sample_image(label: str, filename: str):
    """Serve one bundled example image."""
    from fastapi.responses import FileResponse

    if label not in CLASS_NAMES:
        raise HTTPException(status_code=404, detail="Unknown label.")

    # Reject any path trickery before touching the filesystem — the filename
    # arrives straight from the URL.
    if Path(filename).name != filename or not filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = SAMPLES_DIR / label / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found.")

    return FileResponse(path, media_type="image/png")


@app.get("/metrics")
def metrics() -> dict:
    """Evaluation metrics recorded for the live model."""
    metadata = read_metadata()
    latest = metadata.get("latest")

    if not latest:
        raise HTTPException(
            status_code=404,
            detail="No training events recorded yet. Train the model first.",
        )

    return {
        "model_version": metadata.get("version", 0),
        "recorded_at": latest.get("timestamp"),
        "event": latest.get("event"),
        "metrics": latest.get("metrics", {}),
        "history": [
            {
                "version": index + 1,
                "event": event.get("event"),
                "timestamp": event.get("timestamp"),
                "metrics": event.get("metrics", {}),
            }
            for index, event in enumerate(metadata.get("events", []))
        ],
    }


@app.get("/visualizations")
def visualizations(per_class: int = 400) -> dict:
    """Derived feature distributions powering the dashboard charts.

    Serves the precomputed table shipped with the model when it is available,
    which is always the case in the deployed container — that image carries no
    dataset, so computing features live there is impossible. Falls back to
    computing from raw images on a development machine that has them.
    """
    from src.preprocessing import (
        build_feature_table,
        has_local_images,
        load_feature_table,
    )

    per_class = max(50, min(per_class, 1500))

    frame = load_feature_table()
    source = "precomputed"

    if frame is not None:
        # Subsample per class so the slider still does something meaningful.
        frame = (
            frame.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=42))
            .reset_index(drop=True)
        )
    elif has_local_images():
        frame = build_feature_table(per_class=per_class)
        source = "computed"
    else:
        raise HTTPException(
            status_code=503,
            detail=(
                "No feature data available. The precomputed table "
                "(models/feature_table.csv) is missing and no local images "
                "were found. Run: python -c \"from src.preprocessing import "
                'export_feature_table; export_feature_table()"'
            ),
        )

    features = [
        "intensity_std",
        "dark_pixel_ratio",
        "mean_saturation",
        "mean_intensity",
        "cell_area_ratio",
    ]

    summary = {
        feature: {
            label: {
                "mean": round(float(group[feature].mean()), 5),
                "std": round(float(group[feature].std()), 5),
                "median": round(float(group[feature].median()), 5),
            }
            for label, group in frame.groupby("label")
        }
        for feature in features
    }

    return {
        "sample_size": len(frame),
        "per_class": per_class,
        "source": source,
        "class_counts": frame["label"].value_counts().to_dict(),
        "summary": summary,
        "records": frame.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    """Classify a single uploaded cell image."""
    if not is_model_available():
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run training first.",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict(payload)
    except Exception as error:  # noqa: BLE001 - surfaced to the caller
        raise HTTPException(
            status_code=400, detail=f"Could not process image: {error}"
        ) from error

    result["filename"] = file.filename
    return result


@app.post("/predict/batch")
async def predict_batch_endpoint(files: list[UploadFile] = File(...)) -> dict:
    """Classify several uploaded cell images in one forward pass."""
    if not is_model_available():
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run training first.",
        )

    payloads, filenames = [], []
    for upload in files:
        content = await upload.read()
        if content:
            payloads.append(content)
            filenames.append(upload.filename)

    if not payloads:
        raise HTTPException(status_code=400, detail="No readable images uploaded.")

    try:
        results = predict_batch(payloads)
    except Exception as error:  # noqa: BLE001 - surfaced to the caller
        raise HTTPException(
            status_code=400, detail=f"Could not process images: {error}"
        ) from error

    for filename, result in zip(filenames, results):
        result["filename"] = filename

    counts: dict[str, int] = {}
    for result in results:
        counts[result["prediction"]] = counts.get(result["prediction"], 0) + 1

    return {"count": len(results), "summary": counts, "results": results}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    label: str | None = Form(default=None),
) -> dict:
    """Stage new labelled data for retraining.

    Two shapes are accepted. A `.zip` containing `Parasitized/` and
    `Uninfected/` folders carries its own labels. Any other image file must
    be accompanied by an explicit `label` form field.
    """
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "upload"

    if filename.lower().endswith(".zip"):
        try:
            counts = save_uploaded_archive(payload)
        except Exception as error:  # noqa: BLE001 - surfaced to the caller
            raise HTTPException(
                status_code=400, detail=f"Could not read archive: {error}"
            ) from error

        saved = sum(counts.values())
        if saved == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No labelled images found. The archive must contain "
                    f"{' and '.join(CLASS_NAMES)} folders holding .png images."
                ),
            )
        detail = counts
    else:
        if label not in CLASS_NAMES:
            raise HTTPException(
                status_code=400,
                detail=f"A 'label' field is required, one of {list(CLASS_NAMES)}.",
            )
        saved = save_uploaded_images([(filename, payload)], label=label)
        if saved == 0:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload .png, .jpg or a .zip.",
            )
        detail = {label: saved}

    return {
        "message": f"Staged {saved} image(s) for retraining.",
        "saved": detail,
        "staged_total": count_uploads(),
    }


@app.get("/uploads")
def uploads() -> dict:
    """How much staged data is waiting, and whether it is enough to retrain."""
    counts = count_uploads()
    total = sum(counts.values())
    return {
        "staged": counts,
        "total": total,
        "minimum_required": MIN_IMAGES_TO_RETRAIN,
        "ready_to_retrain": total >= MIN_IMAGES_TO_RETRAIN,
    }


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def _run_retraining(epochs: int, learning_rate: float, include_base_data: bool) -> None:
    """Retrain the deployed model on staged uploads. Runs off the request thread."""
    try:
        import shutil
        import tempfile

        from src.model import evaluate, retrain, write_metadata
        from src.preprocessing import (
            build_datasets,
            build_test_dataset,
            resolve_test_dir,
            resolve_train_dir,
        )

        # Falls back to the images bundled in the container when the full
        # dataset is absent, which is always the case in deployment.
        base_dir_root = resolve_train_dir()
        test_dir_root = resolve_test_dir()

        staged = count_uploads()
        if sum(staged.values()) < MIN_IMAGES_TO_RETRAIN:
            retrain_job.finish(
                "failed",
                f"Need at least {MIN_IMAGES_TO_RETRAIN} staged images, "
                f"found {sum(staged.values())}.",
            )
            return

        # Assemble the training corpus in a temp directory. Mixing the staged
        # uploads with original training data is what stops the model from
        # catastrophically forgetting the baseline distribution when the new
        # batch is small or skewed toward one class.
        #
        # A *sample* of the base data is used rather than all 11,023 images
        # per class. Replaying the entire training set would make one retrain
        # take roughly as long as training from scratch, which defeats the
        # purpose of fine-tuning from existing weights. A few thousand
        # replayed images is enough to anchor the original distribution.
        rng = random.Random(RETRAIN_SEED)

        with tempfile.TemporaryDirectory() as workspace:
            corpus = Path(workspace) / "train"
            corpus_counts: dict[str, int] = {}

            for class_name in CLASS_NAMES:
                (corpus / class_name).mkdir(parents=True, exist_ok=True)
                written = 0

                staged_dir = UPLOAD_DIR / class_name
                if staged_dir.is_dir():
                    for image in staged_dir.iterdir():
                        if image.is_file():
                            shutil.copy2(image, corpus / class_name / image.name)
                            written += 1

                if include_base_data and base_dir_root is not None:
                    base_dir = base_dir_root / class_name
                    if base_dir.is_dir():
                        base_images = [p for p in base_dir.iterdir() if p.is_file()]
                        if len(base_images) > BASE_REPLAY_PER_CLASS:
                            base_images = rng.sample(
                                base_images, BASE_REPLAY_PER_CLASS
                            )
                        for image in base_images:
                            shutil.copy2(
                                image, corpus / class_name / f"base_{image.name}"
                            )
                            written += 1

                corpus_counts[class_name] = written

            print(f"[retrain] training corpus: {corpus_counts}", flush=True)

            train_ds, val_ds = build_datasets(train_dir=corpus)
            model, _ = retrain(
                train_ds, val_ds, epochs=epochs, learning_rate=learning_rate
            )

            if test_dir_root is None:
                retrain_job.finish(
                    "failed",
                    "Retraining ran but no evaluation set is available, so the "
                    "result cannot be scored. The model was not swapped in.",
                )
                return

            scores = evaluate(model, build_test_dataset(test_dir=test_dir_root))
            evaluated_on = sum(
                len(list((test_dir_root / c).glob("*.png"))) for c in CLASS_NAMES
            )

        write_metadata(
            scores,
            event="retrain",
            extra={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "new_images": staged,
                "included_base_data": include_base_data,
                "base_replay_source": str(base_dir_root) if base_dir_root else None,
                "evaluated_on": evaluated_on,
                "evaluation_source": str(test_dir_root),
            },
        )

        # Swap the cached in-memory model for the newly written one, so
        # predictions start using it without a service restart.
        get_model(MODEL_PATH, reload=True)

        retrain_job.finish(
            "completed",
            f"Retraining finished. Test accuracy {scores['accuracy']:.2%}.",
            metrics=scores,
        )

    except Exception as error:  # noqa: BLE001 - recorded for the dashboard
        retrain_job.finish("failed", f"Retraining failed: {error}")


@app.post("/retrain")
def trigger_retrain(
    background_tasks: BackgroundTasks, request: RetrainRequest | None = None
) -> dict:
    """Kick off retraining on the staged uploads and return immediately."""
    settings = request or RetrainRequest()

    staged_total = sum(count_uploads().values())
    if staged_total < MIN_IMAGES_TO_RETRAIN:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough staged data to retrain. Found {staged_total} "
                f"image(s), need at least {MIN_IMAGES_TO_RETRAIN}. "
                "Upload more data first."
            ),
        )

    if not retrain_job.try_claim():
        raise HTTPException(
            status_code=409, detail="A retraining job is already running."
        )

    background_tasks.add_task(
        _run_retraining,
        settings.epochs,
        settings.learning_rate,
        settings.include_base_data,
    )

    return {
        "message": "Retraining started.",
        "staged_images": staged_total,
        "epochs": settings.epochs,
        "learning_rate": settings.learning_rate,
        "poll": "/retrain/status",
    }


@app.get("/retrain/status")
def retrain_status() -> dict:
    """Progress of the running or most recent retrain job."""
    return retrain_job.snapshot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
