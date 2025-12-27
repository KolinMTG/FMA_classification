import pandas as pd
from pathlib import Path
from cste import *
from logger import get_logger
import os
import shutil
import tensorflow as tf
from typing import List
import matplotlib.pyplot as plt
import csv
log = get_logger("others.log")


def document_tracks_header(tracks_csv_path: str, documentation_path: str) -> None:
    """
    Reads a track CSV with multi-line header and writes a documentation
    of its column structure to a text file.

    Args:
        tracks_csv_path: path to tracks.csv
        documentation_path: path to output .txt file
    """
    tracks_csv = Path(tracks_csv_path)
    documentation_file = Path(documentation_path)

    if not tracks_csv.exists():
        raise FileNotFoundError(f"{tracks_csv_path} not found")

    # Read the first 3 lines as header (MultiIndex)
    df = pd.read_csv(tracks_csv, header=[0,1,2])

    # Ensure parent folder exists
    documentation_file.parent.mkdir(parents=True, exist_ok=True)

    with documentation_file.open('w', encoding='utf-8') as f:
        f.write(f"Header structure of {tracks_csv_path}:\n\n")
        f.write("Columns indexed hierarchically:\n\n")

        for idx, col in enumerate(df.columns):
            # col is a tuple representing the multi-level header
            col_levels = [str(c) if str(c) != 'nan' else '' for c in col]
            col_name = " > ".join([level for level in col_levels if level])
            f.write(f"{idx:03d}: {col_name}\n")

    print(f"Header documentation saved to {documentation_file}")



def test_path(path:str)-> bool:
    """Test if a given path exists."""
    p = Path(path)
    return p.exists()

def clear_foler(path:str)-> None:
    """Clear all files in a given folder."""
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"The path {path} does not exist or is not a directory.")
    
    for item in p.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            clear_foler(item)  # Recursively clear subdirectories
            item.rmdir()


def clear_folder(folder: str) -> None:
    """
    Remove all files and directories inside the given folder, without deleting the folder itself.
    """
    folder_path = Path(folder)
    
    if not folder_path.exists() or not folder_path.is_dir():
        log.warning(f"Folder does not exist or is not a directory: {folder_path}")
        return

    for item in folder_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
                log.info(f"Directory removed: {item}")
            else:
                item.unlink()
                log.info(f"File removed: {item}")
        except Exception as e:
            log.error(f"Failed to remove {item}: {e}")

def suppress_useless(suppress_path_list: List[str], historic_name: str = HISTORIC_TRASH_CSV_NAME, target_folder_path: str = TRASH_DIR) -> None:
    """
    Args:
        historic_name (str): Name of the historic CSV file.
        suppress_path_list (List[str]): List of file/directory paths to be moved.
        target_folder_path (str): Path to the target folder where files/directories will be moved.
    Moves specified files/directories to a target folder and logs the operations in a historic CSV file.
    Use this to move all useless data to a separate folder.
    """

    target_folder = Path(target_folder_path)
    target_folder.mkdir(parents=True, exist_ok=True)

    historic_path = target_folder / historic_name
    historic_exists = historic_path.exists()

    with historic_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if not historic_exists:
            writer.writerow(["original_path", "moved_path", "is_dir"])
            log.info(f"Historic file created: {historic_path}")

        for path_str in suppress_path_list:
            original_path = Path(path_str).resolve()

            if not original_path.exists():
                log.info(f"Skipped (not found): {original_path}")
                continue

            relative_path = original_path.relative_to(original_path.anchor)
            moved_path = target_folder / relative_path

            moved_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(original_path), str(moved_path))

            writer.writerow([
                str(original_path),
                str(moved_path),
                original_path.is_dir()
            ])

            if original_path.is_dir():
                log.info(f"Directory moved: {original_path} -> {moved_path}")
            else:
                log.info(f"File moved: {original_path} -> {moved_path}")




def restore_data(historic_name: str = HISTORIC_TRASH_CSV_NAME, source_folder_path: str = TRASH_DIR) -> None:
    """
    Args:
        historic_name (str): Name of the historic CSV file.
        source_folder_path (str): Path to the folder containing the historic file.
    Restores files and directories based on the historic CSV file.
    """
    source_folder = Path(source_folder_path)
    historic_path = source_folder / historic_name

    if not historic_path.exists():
        raise FileNotFoundError("Historic file not found")

    with historic_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        entries = list(reader)

    for entry in entries:
        original_path = Path(entry["original_path"])
        moved_path = Path(entry["moved_path"])
        is_dir = entry["is_dir"] == "True"

        if not moved_path.exists():
            log.info(f"Skipped restore (missing): {moved_path}")
            continue

        original_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(moved_path), str(original_path))

        if is_dir:
            log.info(f"Directory restored: {moved_path} -> {original_path}")
        else:
            log.info(f"File restored: {moved_path} -> {original_path}")

    # Clean up the .trash folder
    clear_folder(source_folder_path)

def suppress_default() -> None:
    """Moves default useless data to the trash folder."""
    suppress_useless(DEFAULT_DATA_TO_SUPPRESS)


def show_spectrogram_from_tfrecord(tfrecord_path: str, index: int = 0) -> None:
    """
    Display a spectrogram from a TFRecord file.

    Args:
        tfrecord_path (str): Path to the TFRecord file containing serialized examples.
        index (int): Index of the example in the TFRecord to display. Defaults to 0.

    The TFRecord is assumed to contain features:
        - 'spectrogram': raw bytes of the spectrogram (float32)
        - 'height': height of the spectrogram
        - 'width': width of the spectrogram
    """
    # Define the feature schema
    feature_description = {
        "spectrogram": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        # 'label' can be parsed too if needed
    }

    # Create dataset from TFRecord
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Parse function
    def _parse_fn(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        height = parsed["height"]
        width = parsed["width"]
        spectrogram = tf.io.decode_raw(parsed["spectrogram"], tf.float32)
        spectrogram = tf.reshape(spectrogram, [height, width])
        return spectrogram

    # Parse all examples
    parsed_dataset = dataset.map(_parse_fn)
    
    # Extract the specified example
    spectrogram = None
    for i, spec in enumerate(parsed_dataset):
        if i == index:
            spectrogram = spec.numpy()
            break

    if spectrogram is None:
        raise IndexError(f"Index {index} out of range for TFRecord {tfrecord_path}")

    # Display the spectrogram
    plt.figure(figsize=(8, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format="%+2.f dB")
    plt.title(f"Spectrogram from {tfrecord_path}, example {index}")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bins")
    plt.show()

def show_class_distribution(filtered_tracks_path:str=FILTERED_TRACK_PATH)-> None:
    """Show class distribution from a filtered tracks CSV file."""
    df = pd.read_csv(filtered_tracks_path)
    genre_counts = df['genre'].value_counts()

    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Tracks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
#     # Example usage
#     # document_tracks_header(
#     #     tracks_csv_path=TRACK_PATH,
#     #     documentation_path=r"documents/tracks_documentation.txt"
#     # )
    # clear_foler(".logs/")
    # clear_foler(r"C:\Users\colin\Documents\ETUDE\MAIN\UTC semestre 5  PK\Neural Networks\final_project\data\tfrecords")
    # show_spectrogram_from_tfrecord(
    #     tfrecord_path=r"data\tfrecords_32\000851.tfrecord",
    #     index=0
    # )

    show_class_distribution()
    pass