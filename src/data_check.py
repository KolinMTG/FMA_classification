import tensorflow as tf
import pandas as pd
import numpy as np
import random
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


from src.cste import *
from src.logger import get_logger

log = get_logger("model_training")

def parse_tfrecord(example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse a TFRecord example into (spectrogram, label) suitable for
    sparse categorical crossentropy training.

    Parameters
    ----------
    example_proto : tf.Tensor
        Serialized TFRecord example.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        spectrogram : Tensor of shape (height, width, channels)
        label : integer Tensor (no one-hot)
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

    # Decode spectrogram bytes and reshape
    spectrogram = tf.io.decode_raw(parsed["spectrogram"], tf.float32)
    spectrogram = tf.reshape(spectrogram, [height, width, NUM_CHANNELS])

    # Label remains an integer for sparse categorical crossentropy
    label = tf.cast(parsed["label"], tf.int32)

    return spectrogram, label

def _collect_split_stats(tfrecord_paths, max_records=None):
    spectro_means = []
    labels = []

    if max_records is not None:
        tfrecord_paths = random.sample(
            tfrecord_paths, min(len(tfrecord_paths), max_records)
        )

    for path in tfrecord_paths:
        dataset = tf.data.TFRecordDataset(path)

        for raw_example in dataset:
            spectrogram, label = parse_tfrecord(raw_example)

            spectro_means.append(tf.reduce_mean(spectrogram).numpy())
            labels.append(label.numpy())

    return np.array(spectro_means), np.array(labels)



def check_normalisation(csv_split_path: str, random_samples: int = 5):
    df = pd.read_csv(csv_split_path)

    split_names = {0: "train", 1: "validation", 2: "test"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, (split_id, split_name) in enumerate(split_names.items()):
        split_df = df[df["split"] == split_id]
        paths = split_df["path"].tolist()

        spectro_means, labels = _collect_split_stats(paths)

        # Histogram of spectrogram means
        axes[0, idx].hist(spectro_means, bins=50)
        axes[0, idx].set_title(f"{split_name} spectrogram mean")
        axes[0, idx].set_xlabel("Mean value")
        axes[0, idx].set_ylabel("Count")

        # Histogram of labels
        axes[1, idx].hist(labels, bins=np.arange(labels.max() + 2) - 0.5)
        axes[1, idx].set_title(f"{split_name} label distribution")
        axes[1, idx].set_xlabel("Label")
        axes[1, idx].set_ylabel("Count")

        print(
            f"{split_name}: mean={spectro_means.mean():.5f}, "
            f"std={spectro_means.std():.5f}"
        )

    plt.tight_layout()
    plt.show()

    # Random TFRecord sanity check
    print("\nRandom TFRecord check")

    for split_id, split_name in split_names.items():
        split_df = df[df["split"] == split_id]
        paths = split_df["path"].tolist()

        spectro_means, _ = _collect_split_stats(
            paths, max_records=random_samples
        )

        print(
            f"{split_name} random sample mean={spectro_means.mean():.5f}, "
            f"std={spectro_means.std():.5f}"
        )


def plot_frequency_distribution(spectrogram: np.ndarray, ax=None, save: str = None):
    freq_mean = np.mean(spectrogram, axis=1)
    if ax is None:
        plt.figure()
        plt.plot(freq_mean)
        plt.title("Frequency Distribution")
        plt.xlabel("Frequency Bin")
        plt.ylabel("Amplitude")
        if save:
            plt.savefig(save + "_freq_dist.png")
        plt.show()
    else:
        ax.plot(freq_mean)
        ax.set_title("Frequency Distribution")
        ax.set_xlabel("Frequency Bin")
        ax.set_ylabel("Amplitude")

def plot_local_structure(spectrogram: np.ndarray, ax=None, save: str = None):
    if ax is None:
        plt.figure()
        plt.imshow(spectrogram[:, :, 0], aspect='auto', origin='lower')
        plt.title("Local Structure")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        if save:
            plt.savefig(save + "_local_structure.png")
        plt.show()
    else:
        ax.imshow(spectrogram[:, :, 0], aspect='auto', origin='lower')
        ax.set_title("Local Structure")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

def plot_temporal_dynamics(spectrogram: np.ndarray, ax=None, save: str = None):
    time_mean = np.mean(spectrogram, axis=0)
    time_mean = time_mean[:, 0] if time_mean.ndim > 1 else time_mean
    if ax is None:
        plt.figure()
        plt.plot(time_mean)
        plt.title("Temporal Dynamics")
        plt.xlabel("Time")
        plt.ylabel("Mean Amplitude")
        if save:
            plt.savefig(save + "_temporal_dynamics.png")
        plt.show()
    else:
        ax.plot(time_mean)
        ax.set_title("Temporal Dynamics")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Amplitude")

def compute_intra_segment_variance(spectrogram: np.ndarray) -> float:
    return np.var(spectrogram)

def inspect_random_spectrogram(csv_split_path: str, split_id=0, num_samples=3, save: str = None):
    df = pd.read_csv(csv_split_path)
    split_df = df[df["split"] == split_id]
    paths = split_df["path"].tolist()
    
    sampled_paths = random.sample(paths, min(num_samples, len(paths)))

    for i, path in enumerate(sampled_paths):
        dataset = tf.data.TFRecordDataset(path)
        for raw_example in dataset.take(1):
            spectrogram, label = parse_tfrecord(raw_example)
            spectrogram = spectrogram.numpy()

            print(f"Label: {label.numpy()}")
            print(f"Intra-segment variance: {compute_intra_segment_variance(spectrogram):.5f}")

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            plot_frequency_distribution(spectrogram, ax=axes[0,0], save=save + f"_sample{i}" if save else None)
            plot_local_structure(spectrogram, ax=axes[0,1], save=save + f"_sample{i}" if save else None)
            plot_temporal_dynamics(spectrogram, ax=axes[1,0], save=save + f"_sample{i}" if save else None)
            axes[1,1].hist(spectrogram.flatten(), bins=50)
            axes[1,1].set_title("Histogram of Values")
            if save:
                plt.savefig(save + f"_sample{i}_hist.png")
            plt.tight_layout()
            plt.show()

def check_intra_segment_variance(csv_split_path: str, max_samples_per_split: int = None):
    """
    Compute and compare intra-segment variance for each split and label.

    Parameters
    ----------
    csv_split_path : str
        Path to CSV containing split information and TFRecord paths.
    max_samples_per_split : int, optional
        Limit the number of TFRecords sampled per split for faster computation.
    """
    df = pd.read_csv(csv_split_path)
    split_names = {0: "train", 1: "validation", 2: "test"}

    results = []

    for split_id, split_name in split_names.items():
        split_df = df[df["split"] == split_id]
        paths = split_df["path"].tolist()

        if max_samples_per_split:
            paths = random.sample(paths, min(len(paths), max_samples_per_split))

        for path in paths:
            dataset = tf.data.TFRecordDataset(path)
            for raw_example in dataset:
                try:
                    spectrogram, label = parse_tfrecord(raw_example)
                    spectrogram = spectrogram.numpy()
                    var = np.var(spectrogram)
                    results.append({
                        "split": split_name,
                        "label": label.numpy(),
                        "variance": var
                    })
                except tf.errors.OutOfRangeError:
                    break

    # Convert results to DataFrame for easier analysis
    var_df = pd.DataFrame(results)

    # Summary statistics per split
    print("Variance summary per split:")
    print(var_df.groupby("split")["variance"].describe())

    # Summary statistics per label
    print("\nVariance summary per label:")
    print(var_df.groupby("label")["variance"].describe())

    # Optional: boxplot visualization
    plt.figure(figsize=(12,5))
    sns.boxplot(x="split", y="variance", data=var_df)
    plt.title("Intra-segment Variance per Split")
    plt.show()

    plt.figure(figsize=(12,5))
    sns.boxplot(x="label", y="variance", data=var_df)
    plt.title("Intra-segment Variance per Label")
    plt.show()

    return var_df


def filter_binary_classes(csv_input_path: str, csv_output_path: str):
    """
    Keep only samples with label 0 or 1 and save a new CSV.

    Parameters
    ----------
    csv_input_path : str
        Path to the original CSV containing 'split', 'path', 'label'.
    csv_output_path : str
        Path where the filtered CSV will be saved.
    """
    df = pd.read_csv(csv_input_path)

    # Keep only rows where label is 0 or 1
    df_binary = df[df["label"].isin([0, 1])].copy()

    # Save the filtered CSV
    df_binary.to_csv(csv_output_path, index=False)
    print(f"Filtered CSV saved to {csv_output_path}, containing {len(df_binary)} samples.")

    return df_binary




# =========================
# Execution example
# =========================
if __name__ == "__main__":
    csv_split_path = DATA_SPLIT_CSV_PATH_32
    # check_normalisation(csv_split_path, random_samples=5)
    # inspect_random_spectrogram(csv_split_path, split_id=0, num_samples=3, save=SAVE_PLOT_PATH + "train")
    # inspect_random_spectrogram(csv_split_path, split_id=1, num_samples=3, save=SAVE_PLOT_PATH + "validation")
    # inspect_random_spectrogram(csv_split_path, split_id=2, num_samples=3, save=SAVE_PLOT_PATH + "test")
    # check_intra_segment_variance(csv_split_path, max_samples_per_split=10)
    # binary_mapping = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1}  # Example mapping
    df_binary = filter_binary_classes(csv_split_path, "data/metadata/dataset_split_binary_32.csv")

