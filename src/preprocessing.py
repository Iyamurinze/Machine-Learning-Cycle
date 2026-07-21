"""Preprocessing for the malaria cell classification pipeline.

Two responsibilities live here:

1. **Model-facing preprocessing** — turning directories of PNGs into batched,
   normalized, augmented `tf.data` pipelines, and turning a single uploaded
   image into a model-ready tensor.

2. **Analysis-facing feature extraction** — deriving interpretable scalar
   features from each cell image. The NIH images are raw pixels with no
   accompanying metadata, so the features used for exploratory analysis and
   for the dashboard visualizations are computed here rather than read from
   a CSV. See `extract_features` for what each one means clinically.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"

# 64x64 is a deliberate choice, not a default. The NIH images are already
# segmented single cells averaging ~130px, and the parasite chromatin dot
# survives downsampling to 64px while training stays fast enough to retrain
# inside a web request on CPU-only cloud hardware.
IMG_SIZE = (64, 64)
CLASS_NAMES = ("Parasitized", "Uninfected")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Pixels this dark are the black slide background the NIH segmentation left
# around each cell, not cell material.
BACKGROUND_THRESHOLD = 30


# ---------------------------------------------------------------------------
# Model-facing preprocessing
# ---------------------------------------------------------------------------

def build_datasets(
    train_dir: Path = TRAIN_DIR,
    validation_split: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
):
    """Build shuffled, prefetched train/validation datasets from a directory.

    Returns `(train_ds, val_ds)` with labels as float32 binary targets and
    pixels left in [0, 255] — rescaling happens inside the model so that the
    exact same normalization is applied at serving time without the API
    having to remember to do it.
    """
    import tensorflow as tf

    common = dict(
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        seed=seed,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=validation_split, subset="training", **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=validation_split, subset="validation", **common
    )

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune),
        val_ds.cache().prefetch(autotune),
    )


def build_test_dataset(test_dir: Path = TEST_DIR, batch_size: int = 64):
    """Load the held-out test set unshuffled so predictions align to labels."""
    import tensorflow as tf

    return tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )


def build_augmentation():
    """Augmentation applied only during training.

    Blood cells have no canonical orientation — a parasitized cell is still
    parasitized upside down — so flips and rotations are label-preserving
    here in a way they would not be for, say, digit recognition. This is the
    main defense against overfitting given how visually uniform the classes
    are.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def preprocess_image(source: str | Path | bytes | Image.Image) -> np.ndarray:
    """Turn one image into a model-ready batch of shape (1, H, W, 3).

    Accepts a path, raw bytes (as uploaded through the API), or a PIL image.
    """
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)

    image = image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


# ---------------------------------------------------------------------------
# Analysis-facing feature extraction
# ---------------------------------------------------------------------------

def _cell_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels belonging to the cell rather than the background."""
    return rgb.max(axis=2) > BACKGROUND_THRESHOLD


def _saturation(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel HSV saturation, computed directly to avoid a cv2 dependency."""
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    # Guard the divide: pure-black background pixels have high == 0.
    return np.where(high > 0, (high - low) / np.maximum(high, 1), 0.0)


def extract_features(source: str | Path | Image.Image) -> dict[str, float]:
    """Derive interpretable scalar features from a single cell image.

    The three features the analysis leans on, ordered by the class
    separation they actually achieve (Cohen's d measured on a 1,200-image
    sample, reported in the notebook):

    `intensity_std`  (d = 1.12, large)
        Texture heterogeneity within the cell. A healthy red blood cell is a
        smooth, uniform disc; an invaded one contains a chromatin dot plus
        disturbed cytoplasm, so its pixel intensities scatter far more
        widely. This is the single strongest hand-crafted signal.

    `dark_pixel_ratio`  (d = 0.79, medium-large)
        Fraction of cell pixels markedly darker than that cell's own median
        brightness — i.e. how much dense parasite chromatin is present.
        Parasitized cells average 0.90% against 0.03% for uninfected, a 30x
        gap. Measuring against each cell's own median rather than a fixed
        cutoff keeps it robust to slide-to-slide exposure differences.

    `mean_saturation`  (d = 0.57, medium)
        Giemsa stain binds parasite DNA and turns it deep purple, against
        the flat desaturated pink of a healthy cell.

    Two further features are returned for the notebook's correlation
    analysis. `mean_intensity` separates the classes weakly (d = -0.49).
    `cell_area_ratio` does not meaningfully separate them at all (d = -0.20)
    and is retained as a deliberate negative result: the NIH pipeline
    segments and crops each cell tightly, which normalizes away the very
    size-and-shape variation this feature was intended to capture.
    """
    image = source if isinstance(source, Image.Image) else Image.open(source)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

    mask = _cell_mask(rgb)
    cell_pixels = rgb[mask]

    # A fully-black frame would otherwise produce NaNs downstream.
    if cell_pixels.size == 0:
        return {
            "mean_saturation": 0.0,
            "dark_pixel_ratio": 0.0,
            "cell_area_ratio": 0.0,
            "mean_intensity": 0.0,
            "intensity_std": 0.0,
        }

    intensity = cell_pixels.mean(axis=1)
    median_intensity = float(np.median(intensity))

    return {
        "mean_saturation": float(_saturation(rgb)[mask].mean()),
        "dark_pixel_ratio": float((intensity < median_intensity * 0.75).mean()),
        "cell_area_ratio": float(mask.mean()),
        "mean_intensity": float(intensity.mean()),
        "intensity_std": float(intensity.std()),
    }


# Precomputed feature distributions, shipped next to the model.
#
# The deployed container deliberately does not carry the 27k-image dataset —
# it is a gigabyte, and a serving image has no use for it. But the dashboard's
# insight charts describe that dataset, so the derived features are computed
# once here and committed. This also makes the endpoint instant rather than
# sweeping hundreds of images per request.
FEATURE_TABLE_PATH = PROJECT_ROOT / "models" / "feature_table.csv"


def export_feature_table(
    destination: Path = FEATURE_TABLE_PATH,
    per_class: int = 1500,
    seed: int = 42,
):
    """Compute the feature table from local images and persist it to CSV."""
    frame = build_feature_table(per_class=per_class, seed=seed)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"[features] wrote {len(frame):,} rows to {destination}")
    return frame


def load_feature_table(source: Path = FEATURE_TABLE_PATH):
    """Load the precomputed feature table, or None if it has not been built."""
    import pandas as pd

    source = Path(source)
    if not source.exists():
        return None
    return pd.read_csv(source)


def has_local_images(train_dir: Path = TRAIN_DIR) -> bool:
    """Whether raw training images are present on this machine."""
    return any(
        (Path(train_dir) / class_name).is_dir()
        and any((Path(train_dir) / class_name).glob("*.png"))
        for class_name in CLASS_NAMES
    )


# A small train/test pair shipped inside the container.
#
# The full 27k dataset is deliberately excluded from the image, but retraining
# still needs two things from it: original images to replay so fine-tuning
# does not catastrophically forget the baseline, and a held-out set to
# evaluate the result. Without these the retrain endpoint trains on the
# uploaded batch alone and then dies trying to evaluate against a directory
# that does not exist.
#
# 300 images per class per split, disjoint from each other and from the
# prediction samples. ~16 MB.
BUNDLED_TRAIN_DIR = DATA_DIR / "bundled" / "train"
BUNDLED_TEST_DIR = DATA_DIR / "bundled" / "test"


def _has_images(directory: Path) -> bool:
    return any(
        (Path(directory) / class_name).is_dir()
        and any((Path(directory) / class_name).glob("*.png"))
        for class_name in CLASS_NAMES
    )


def resolve_train_dir() -> Path | None:
    """Training images to replay: the full dataset if present, else bundled."""
    if _has_images(TRAIN_DIR):
        return TRAIN_DIR
    if _has_images(BUNDLED_TRAIN_DIR):
        return BUNDLED_TRAIN_DIR
    return None


def resolve_test_dir() -> Path | None:
    """Evaluation images: the full held-out set if present, else bundled."""
    if _has_images(TEST_DIR):
        return TEST_DIR
    if _has_images(BUNDLED_TEST_DIR):
        return BUNDLED_TEST_DIR
    return None


def build_feature_table(
    directories: Iterable[Path] | None = None,
    per_class: int | None = 1500,
    seed: int = 42,
):
    """Build a tidy DataFrame of derived features labelled by class.

    `per_class` caps how many images are sampled from each class — the full
    27k-image sweep takes minutes and adds nothing to the distributions.
    Pass `None` to use everything.
    """
    import pandas as pd

    directories = list(directories) if directories else [TRAIN_DIR]
    rng = np.random.default_rng(seed)
    rows = []

    for class_name in CLASS_NAMES:
        paths: list[Path] = []
        for directory in directories:
            class_dir = Path(directory) / class_name
            if class_dir.is_dir():
                paths.extend(
                    p for p in class_dir.iterdir()
                    if p.suffix.lower() in VALID_EXTENSIONS
                )

        if per_class is not None and len(paths) > per_class:
            picks = rng.choice(len(paths), size=per_class, replace=False)
            paths = [paths[i] for i in picks]

        for path in paths:
            features = extract_features(path)
            features["label"] = class_name
            rows.append(features)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Upload handling for retraining
# ---------------------------------------------------------------------------

def save_uploaded_archive(
    archive_bytes: bytes, destination: Path = UPLOAD_DIR
) -> dict[str, int]:
    """Unpack a user-uploaded zip of new labelled images into the upload store.

    The archive is expected to contain `Parasitized/` and `Uninfected/`
    folders. Entries are validated on the way out rather than blanket
    extracted, which both enforces the label structure and prevents a
    malicious archive from writing outside the destination directory.
    """
    destination = Path(destination)
    counts = {name: 0 for name in CLASS_NAMES}

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            path = Path(member.filename)
            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            # Match the label folder anywhere in the path so both
            # "Parasitized/x.png" and "batch/Parasitized/x.png" work.
            label = next(
                (c for c in CLASS_NAMES if c.lower() in
                 [part.lower() for part in path.parts]),
                None,
            )
            if label is None:
                continue

            target_dir = destination / label
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / path.name

            with archive.open(member) as source, open(target, "wb") as sink:
                sink.write(source.read())
            counts[label] += 1

    return counts


def save_uploaded_images(
    files: Iterable[tuple[str, bytes]], label: str, destination: Path = UPLOAD_DIR
) -> int:
    """Save individually uploaded images under a chosen label."""
    if label not in CLASS_NAMES:
        raise ValueError(f"label must be one of {CLASS_NAMES}, got {label!r}")

    target_dir = Path(destination) / label
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for filename, payload in files:
        if Path(filename).suffix.lower() not in VALID_EXTENSIONS:
            continue
        (target_dir / Path(filename).name).write_bytes(payload)
        saved += 1

    return saved


def count_uploads(destination: Path = UPLOAD_DIR) -> dict[str, int]:
    """Count staged upload images per class, for the dashboard and retrain gate."""
    destination = Path(destination)
    return {
        name: sum(
            1 for p in (destination / name).iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        )
        if (destination / name).is_dir()
        else 0
        for name in CLASS_NAMES
    }
