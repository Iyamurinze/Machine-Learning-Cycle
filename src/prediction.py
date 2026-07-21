"""Inference for the malaria cell classification pipeline.

Serving-time concerns live here: loading the model once and keeping it warm,
turning a single uploaded image into a labelled prediction, and scoring
batches.

A note on label polarity, because it is the easiest thing in this project to
get silently backwards. The datasets are built with
`class_names=["Parasitized", "Uninfected"]` and `label_mode="binary"`, so
Keras assigns:

    Parasitized -> 0
    Uninfected  -> 1

The network ends in a single sigmoid, so its raw output is P(Uninfected).
A cell is therefore predicted parasitized when the sigmoid output is *below*
the threshold. `POSITIVE_CLASS_INDEX` encodes this in one place so the API,
the UI and the notebook cannot drift apart on it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np

from src.model import MODEL_PATH, load_model
from src.preprocessing import CLASS_NAMES, preprocess_image

# Index into CLASS_NAMES that the sigmoid output represents the probability of.
POSITIVE_CLASS_INDEX = 1  # -> "Uninfected"

DEFAULT_THRESHOLD = 0.5

_MODEL = None
_MODEL_PATH: Path | None = None
_LOADED_AT: float | None = None


def get_model(model_path: Path = MODEL_PATH, reload: bool = False):
    """Return the cached model, loading it on first use.

    Keras model loading takes seconds, which would dominate the latency of
    every request and wreck the Locust numbers if it happened per call. The
    cache is invalidated automatically when `model_path` changes and can be
    forced with `reload=True` after a retrain swaps the file underneath us.
    """
    global _MODEL, _MODEL_PATH, _LOADED_AT

    model_path = Path(model_path)
    if _MODEL is None or reload or _MODEL_PATH != model_path:
        _MODEL = load_model(model_path)
        _MODEL_PATH = model_path
        _LOADED_AT = time.time()

    return _MODEL


def model_loaded_at() -> float | None:
    """Timestamp of the current model load, for the dashboard's uptime panel."""
    return _LOADED_AT


def is_model_available(model_path: Path = MODEL_PATH) -> bool:
    """Whether a trained model exists on disk, without loading it."""
    return Path(model_path).exists()


def _label_from_probability(
    probability_positive: float, threshold: float
) -> tuple[str, float]:
    """Map a sigmoid output to (label, confidence in that label)."""
    if probability_positive >= threshold:
        return CLASS_NAMES[POSITIVE_CLASS_INDEX], probability_positive
    return CLASS_NAMES[1 - POSITIVE_CLASS_INDEX], 1.0 - probability_positive


def predict(
    source: str | Path | bytes,
    model=None,
    threshold: float = DEFAULT_THRESHOLD,
    model_path: Path = MODEL_PATH,
) -> dict:
    """Classify a single cell image.

    Accepts a file path or raw bytes as uploaded through the API. Returns the
    predicted label, the model's confidence in that label, the explicit
    per-class probabilities, and the inference latency.
    """
    model = model if model is not None else get_model(model_path)

    started = time.perf_counter()
    batch = preprocess_image(source)
    probability_uninfected = float(model.predict(batch, verbose=0)[0][0])
    latency_ms = (time.perf_counter() - started) * 1000

    label, confidence = _label_from_probability(probability_uninfected, threshold)

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[0]: round(1.0 - probability_uninfected, 4),
            CLASS_NAMES[1]: round(probability_uninfected, 4),
        },
        "threshold": threshold,
        "latency_ms": round(latency_ms, 2),
    }


def predict_batch(
    sources: Iterable[str | Path | bytes],
    model=None,
    threshold: float = DEFAULT_THRESHOLD,
    model_path: Path = MODEL_PATH,
) -> list[dict]:
    """Classify several images in one forward pass.

    Batching matters for throughput: one `predict` call over 32 images is
    substantially cheaper than 32 separate calls, since per-call Keras
    overhead is amortized.
    """
    sources = list(sources)
    if not sources:
        return []

    model = model if model is not None else get_model(model_path)

    batch = np.concatenate([preprocess_image(source) for source in sources], axis=0)

    started = time.perf_counter()
    probabilities = model.predict(batch, verbose=0).reshape(-1)
    latency_ms = (time.perf_counter() - started) * 1000

    results = []
    for probability_uninfected in probabilities:
        probability_uninfected = float(probability_uninfected)
        label, confidence = _label_from_probability(probability_uninfected, threshold)
        results.append(
            {
                "prediction": label,
                "confidence": round(confidence, 4),
                "probabilities": {
                    CLASS_NAMES[0]: round(1.0 - probability_uninfected, 4),
                    CLASS_NAMES[1]: round(probability_uninfected, 4),
                },
            }
        )

    # Report the amortized per-image cost rather than the whole-batch time.
    per_image_ms = round(latency_ms / len(results), 2)
    for result in results:
        result["latency_ms"] = per_image_ms

    return results


def predict_proba(
    source: str | Path | bytes, model=None, model_path: Path = MODEL_PATH
) -> float:
    """Raw sigmoid output, i.e. P(Uninfected). Useful for ROC analysis."""
    model = model if model is not None else get_model(model_path)
    return float(model.predict(preprocess_image(source), verbose=0)[0][0])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a single cell image.")
    parser.add_argument("image", help="path to a cell image")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    outcome = predict(args.image, threshold=args.threshold)
    print(f"Prediction : {outcome['prediction']}")
    print(f"Confidence : {outcome['confidence']:.2%}")
    print(f"Latency    : {outcome['latency_ms']} ms")
