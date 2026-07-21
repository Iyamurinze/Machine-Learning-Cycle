"""Model definition, training and retraining for malaria cell classification.

The architecture is a purpose-built CNN rather than a fine-tuned ImageNet
backbone. Segmented single-cell microscopy looks nothing like ImageNet's
natural images, the discriminating signal is a small stained blob rather than
object shape, and a compact custom network reaches ~96-97% here while staying
small enough to retrain on CPU-only cloud hardware inside a request.

Optimization techniques applied (all four are exercised in the notebook):
  * Regularization  — L2 weight decay, dropout, batch normalization
  * Optimizer       — Adam with an explicit, tunable learning rate
  * Early stopping  — halts on stalled validation loss and restores the best weights
  * LR scheduling   — ReduceLROnPlateau to refine once progress flattens

`retrain` deliberately loads the previously saved model and continues training
it at a reduced learning rate. That is the "use your own model as a pretrained
model" requirement: the deployed model is the starting point, not a fresh
random initialization.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "malaria_cnn.keras"
METADATA_PATH = MODELS_DIR / "metadata.json"

IMG_SIZE = (64, 64)
INPUT_SHAPE = (*IMG_SIZE, 3)


def build_model(
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    l2_factor: float = 1e-4,
    filters: tuple[int, ...] = (32, 64, 128),
):
    """Build and compile the CNN.

    Three convolutional blocks of increasing width, each Conv -> BN -> Conv ->
    BN -> MaxPool -> Dropout, then global average pooling instead of a large
    flatten. Global pooling is what keeps the parameter count near 100k rather
    than several million, which is most of why this model resists overfitting
    on a visually homogeneous dataset.

    Every argument here is a hyperparameter the notebook tunes over.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    from src.preprocessing import build_augmentation

    regularizer = regularizers.l2(l2_factor)

    model = tf.keras.Sequential(name="malaria_cnn")
    model.add(layers.Input(shape=INPUT_SHAPE))

    # Augmentation and rescaling live inside the model so that serving code
    # only ever has to hand over raw [0, 255] pixels. Augmentation layers are
    # automatically inert at inference time.
    model.add(build_augmentation())
    model.add(layers.Rescaling(1.0 / 255))

    for block_index, filter_count in enumerate(filters):
        for _ in range(2):
            model.add(
                layers.Conv2D(
                    filter_count,
                    3,
                    padding="same",
                    activation="relu",
                    kernel_regularizer=regularizer,
                )
            )
            model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        # Dropout ramps up with depth: light early where features are generic,
        # heavier later where the network is most prone to memorizing.
        model.add(layers.Dropout(dropout_rate * (0.5 + 0.25 * block_index)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(
        layers.Dense(64, activation="relu", kernel_regularizer=regularizer)
    )
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_callbacks(
    checkpoint_path: Path | None = None,
    patience: int = 5,
    monitor: str = "val_loss",
):
    """Early stopping, LR scheduling, and optional best-weights checkpointing."""
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor=monitor,
                save_best_only=True,
                verbose=0,
            )
        )

    return callbacks


def train(
    train_ds,
    val_ds,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    l2_factor: float = 1e-4,
    model_path: Path = MODEL_PATH,
    save: bool = True,
):
    """Train a fresh model from random initialization. Returns (model, history)."""
    model = build_model(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_factor=l2_factor,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=build_callbacks(),
        verbose=1,
    )

    if save:
        save_model(model, model_path)

    return model, history


def retrain(
    train_ds,
    val_ds,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    model_path: Path = MODEL_PATH,
    save: bool = True,
):
    """Continue training the saved model on new data.

    The learning rate defaults to a tenth of the from-scratch rate. Loading
    trained weights and then hitting them with the original learning rate
    would wash out what the model already knows; a lower rate adapts to the
    new images while preserving the existing decision boundary.

    Falls back to training from scratch only if no saved model exists yet.
    """
    import tensorflow as tf

    model_path = Path(model_path)

    if not model_path.exists():
        print(f"[retrain] no existing model at {model_path}; training from scratch")
        return train(train_ds, val_ds, epochs=epochs, model_path=model_path, save=save)

    print(f"[retrain] loading pretrained model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Recompile to install the reduced learning rate while keeping the
    # existing weights and metric set.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=build_callbacks(patience=3),
        verbose=1,
    )

    if save:
        save_model(model, model_path)

    return model, history


def evaluate(model, test_ds) -> dict[str, float]:
    """Evaluate on the held-out test set, returning every rubric metric.

    Keras gives accuracy, loss, precision, recall and AUC directly; F1 is
    derived from precision and recall since Keras has no built-in binary F1.
    """
    import numpy as np

    results = model.evaluate(test_ds, verbose=0, return_dict=True)

    precision = float(results.get("precision", 0.0))
    recall = float(results.get("recall", 0.0))
    denominator = precision + recall

    metrics = {
        "accuracy": float(results.get("accuracy", 0.0)),
        "loss": float(results.get("loss", 0.0)),
        "precision": precision,
        "recall": recall,
        "auc": float(results.get("auc", 0.0)),
        "f1_score": float(2 * precision * recall / denominator) if denominator else 0.0,
    }

    return {key: round(value, 4) for key, value in metrics.items()}


def save_model(model, model_path: Path = MODEL_PATH) -> Path:
    """Persist the model to the native Keras format."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"[save] model written to {model_path}")
    return model_path


def load_model(model_path: Path = MODEL_PATH):
    """Load a saved model, raising a clear error if training never ran."""
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Run the training notebook or "
            f"src/model.py first."
        )
    return tf.keras.models.load_model(model_path)


def write_metadata(
    metrics: dict[str, float],
    event: str = "train",
    extra: dict | None = None,
    metadata_path: Path = METADATA_PATH,
) -> dict:
    """Append a training event to the model's history log.

    The dashboard reads this to show which model version is live, when it was
    last retrained, and how production metrics have moved across retrainings.
    """
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        history = json.loads(metadata_path.read_text())
    else:
        history = {"events": []}

    entry = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        **(extra or {}),
    }
    history["events"].append(entry)
    history["latest"] = entry
    history["version"] = len(history["events"])

    metadata_path.write_text(json.dumps(history, indent=2))
    return history


def read_metadata(metadata_path: Path = METADATA_PATH) -> dict:
    """Read the training history log, or an empty log if none exists."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {"events": [], "latest": None, "version": 0}
    return json.loads(metadata_path.read_text())
