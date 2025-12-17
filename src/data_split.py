# data_split.py

from pathlib import Path
import pandas as pd
import numpy as np
from src.logger import get_logger
import tensorflow as tf
from src.cste import Split

log = get_logger("data_split.log")

# Enum pour splits plus reproductibles


def get_label_from_tfrecord(tfrecord_path: str) -> int:
    """
    Extract the label from a TFRecord file.
    Only reads the first example for efficiency.
    """
    try:
        raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
        feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        for raw_record in raw_dataset.take(1):
            example = tf.io.parse_single_example(raw_record, feature_description)
            return int(example['label'].numpy())
    except Exception as e:
        log.warning(f"Unable to read label from {tfrecord_path}: {e}")
        return None


def split_indices(num_samples: int, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """
    Generate train/val/test indices.
    """
    np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx


def data_split_pipeline(tfrecord_folder: str,
                        output_csv_path: str,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        seed: int = 42) -> bool:
    """
    Complete pipeline to split TFRecord dataset into train/val/test CSV.

    Parameters
    ----------
    tfrecord_folder : str
        Folder containing all TFRecord files.
    output_csv_path : str
        Path to save CSV containing columns: path,label,split
    train_ratio : float
        Ratio of train dataset.
    val_ratio : float
        Ratio of validation dataset.
    test_ratio : float
        Ratio of test dataset.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    bool
        True if CSV successfully created, False otherwise.
    """
    try:
        tfrecord_folder = Path(tfrecord_folder)
        if not tfrecord_folder.exists():
            log.error(f"TFRecord folder not found: {tfrecord_folder}")
            return False

        # List all .tfrecord files
        tfrecord_paths = sorted(tfrecord_folder.glob("*.tfrecord"))
        if len(tfrecord_paths) == 0:
            log.error("No TFRecord files found in folder.")
            return False

        log.info(f"Found {len(tfrecord_paths)} TFRecord files in {tfrecord_folder}")

        # Extract labels for all TFRecords
        labels = []
        for path in tfrecord_paths:
            # For simplicity, labels can be inferred from filename or TFRecord itself
            # Here we assume label is encoded in the TFRecord
            label = get_label_from_tfrecord(str(path))
            if label is None:
                log.warning(f"Skipping {path} because label could not be read.")
                continue
            labels.append(label)

        if len(labels) == 0:
            log.error("No valid TFRecords with labels found.")
            return False

        tfrecord_paths = [str(p) for p, l in zip(tfrecord_paths, labels)]
        labels = np.array(labels)

        # Generate train/val/test splits
        train_idx, val_idx, test_idx = split_indices(len(tfrecord_paths),
                                                     train_ratio=train_ratio,
                                                     val_ratio=val_ratio,
                                                     seed=seed)

        split_values = np.zeros(len(tfrecord_paths), dtype=int)
        split_values[val_idx] = Split.VAL
        split_values[test_idx] = Split.TEST
        split_values[train_idx] = Split.TRAIN

        # Build dataframe
        df_split = pd.DataFrame({
            'path': tfrecord_paths,
            'label': labels,
            'split': split_values
        })

        # Save CSV
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_split.to_csv(output_csv_path, index=False)
        log.info(f"CSV with dataset split saved at {output_csv_path}")

        return True

    except Exception as e:
        log.error(f"Failed to create dataset split CSV: {e}")
        return False
    
if __name__ == "__main__":
    # Example usage
    success = data_split_pipeline(
        tfrecord_folder="data/tfrecords",
        output_csv_path="data/metadata/dataset_split.csv",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    if success:
        log.info("Data split pipeline completed successfully.")
    else:
        log.error("Data split pipeline failed.")
