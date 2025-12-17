import os
from pathlib import Path
import tensorflow as tf
from typing import List, Tuple
from datetime import datetime
import pandas as pd
from src.cste import *
from src.logger import get_logger

log = get_logger("model_training")

# --------------------------------------------------
# TFRecord parsing
# --------------------------------------------------

def parse_tfrecord(example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse a TFRecord example into (spectrogram, label).
    """
    features = {
        "spectrogram": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
    }

    parsed = tf.io.parse_single_example(example_proto, features)

    height = parsed["height"]
    width = parsed["width"]

    spectrogram = tf.io.decode_raw(parsed["spectrogram"], tf.float32)
    spectrogram = tf.reshape(spectrogram, [height, width, NUM_CHANNELS])

    label = tf.cast(parsed["label"], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)

    return spectrogram, label

# --------------------------------------------------
# Dataset builder
# --------------------------------------------------


def build_dataset_from_csv(
    tfrecord_dir: str = TFRECORD_OUTPUT_DIR,
    csv_path: str = DATA_SPLIT_CSV_PATH,
    split_value: int = SplitLabels.TRAIN,
    batch_size: int = TrainingConstants.BATCH_SIZE,
    shuffle: bool = False,
):
    """Construct a tf.data.Dataset from TFRecord files listed in a CSV file, only including those with a specific split value. 0 = Train, 1 = Val, 2 = Test."""
    df = pd.read_csv(csv_path)

    # Filter by split
    df = df[df["split"] == split_value]

    # Build full paths
    tfrecord_paths = [
        str(Path(tfrecord_dir) / Path(p).name) for p in df["path"]
    ]

    dataset = tf.data.TFRecordDataset(tfrecord_paths)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(tfrecord_paths))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def train_model(
    model: tf.keras.Model,
    tfrecord_dir: str,
    csv_path: str,
    batch_size: int = TrainingConstants.BATCH_SIZE,
    epochs: int = TrainingConstants.EPOCHS,
    learning_rate: float = TrainingConstants.LEARNING_RATE,
    early_stopping_patience: int = 5,
    save: bool = False,
    model_save_dir: str | None = ModelsCSV.TRAINING,
    model_registry_csv: str | None = ModelsCSV.REGISTRY,
    notes: str = "",
):
    """
    Train a TensorFlow model using TFRecords and optionally save it.

    Parameters
    ----------
    save : bool
        If True, the trained model is saved and logged to CSV.
    """

    train_ds = build_dataset_from_csv(
        tfrecord_dir,
        csv_path,
        split_value=0,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = build_dataset_from_csv(
        tfrecord_dir,
        csv_path,
        split_value=1,
        batch_size=batch_size,
        shuffle=False,
    )

    # if not model._is_compiled:
    #     model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    #         loss="sparse_categorical_crossentropy",
    #         metrics=["accuracy"],
    #     )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Optional save
    if save:
        if not all([model_save_dir, model.name, model_registry_csv]):
            raise ValueError(
                "model_save_dir, model_name and model_registry_csv must be provided when save=True"
            )

        best_val_loss = min(history.history["val_loss"])
        best_val_accuracy = max(history.history.get("val_accuracy", []))
        epochs_trained = len(history.history["loss"])

        save_model_and_log(
            model=model,
            model_save_dir=model_save_dir,
            model_name=model.name,
            csv_log_path=model_registry_csv,
            dataset_csv_path=csv_path,
            epochs_trained=epochs_trained,
            batch_size=batch_size,
            val_loss=best_val_loss,
            val_accuracy=best_val_accuracy,
            notes=notes,
        )

    return model




def save_model_and_log(
    model: tf.keras.Model,
    model_save_dir: str,
    model_name: str,
    csv_log_path: str,
    dataset_csv_path: str,
    epochs_trained: int = TrainingConstants.EPOCHS,
    batch_size: int = TrainingConstants.BATCH_SIZE,
    val_loss: float | None = None,
    val_accuracy: float | None = None,
    notes: str = "",
):
    """
    Save a trained TensorFlow model and log metadata into a CSV file.

    Parameters
    ----------
    model : tf.keras.Model
        Trained TensorFlow model.
    model_save_dir : str
        Directory where the model will be saved.
    model_name : str
        Human-readable model name.
    csv_log_path : str
        Path to the CSV file storing model metadata.
    dataset_csv_path : str
        Path to the dataset split CSV used for training.
    epochs_trained : int
        Number of epochs actually trained.
    batch_size : int
        Training batch size.
    val_loss : float, optional
        Validation loss of the best model.
    val_accuracy : float, optional
        Validation accuracy of the best model.
    notes : str
        Free text notes.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_save_dir / f"{model_name}_{timestamp}"
    model.save(model_path)

    log_entry = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_path": str(model_path),
        "dataset_csv_path": dataset_csv_path,
        "epochs_trained": epochs_trained,
        "batch_size": batch_size,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "framework": "tensorflow",
        "notes": notes,
    }

    csv_log_path = Path(csv_log_path)
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_log_path.exists():
        df = pd.read_csv(csv_log_path)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(csv_log_path, index=False)

