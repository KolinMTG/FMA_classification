import os
from pathlib import Path
import tensorflow as tf
from typing import List, Tuple

from src.cste import *
from src.models.model import build_cnn_model
from src.logger import get_logger

log = get_logger("model_training")
AUTOTUNE = tf.data.AUTOTUNE

# --------------------------------------------------
# TFRecord parsing
# --------------------------------------------------

def parse_tfrecord(example_proto):
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

def load_tfrecord_files(tfrecord_dir: str = TFRECORD_DIR) -> List[str]:
    """
    Load all TFRecord files from directory.
    """
    files = sorted(str(f) for f in Path(tfrecord_dir).glob("*.tfrecord"))
    if not files:
        raise RuntimeError("No TFRecord files found")
    return files

def split_files(files: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split TFRecord files into train / validation / test.
    """
    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    return (
        files[:train_end],
        files[train_end:val_end],
        files[val_end:]
    )

def build_dataset(files: List[str], training: bool) -> tf.data.Dataset:
    """
    Build tf.data.Dataset from TFRecord files.
    """
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

# --------------------------------------------------
# Training pipeline
# --------------------------------------------------

def train_pipeline():
    """
    Full training pipeline.
    """
    log.info("Loading TFRecord files")
    files = load_tfrecord_files(TFRECORD_DIR)

    train_files, val_files, test_files = split_files(files)

    log.info(f"Train files: {len(train_files)}")
    log.info(f"Validation files: {len(val_files)}")
    log.info(f"Test files: {len(test_files)}")

    train_ds = build_dataset(train_files, training=True)
    val_ds = build_dataset(val_files, training=False)
    test_ds = build_dataset(test_files, training=False)

    # Infer input shape dynamically
    for x, _ in train_ds.take(1):
        input_shape = x.shape[1:]

    log.info(f"Model input shape: {input_shape}")

    model = build_cnn_model(input_shape)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    log.info("Starting training")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    log.info("Evaluating on test set")
    loss, acc = model.evaluate(test_ds)
    log.info(f"Test accuracy: {acc:.4f}")
    log.info(f"Test loss: {loss:.4f}")

if __name__ == "__main__":
    train()
