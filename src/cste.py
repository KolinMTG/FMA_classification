# This file contains all constants used throughout the project
from pathlib import Path
from typing import List

# PC configuration
NUM_WORKERS : int = 4  # Number of parallel workers for data processing

# LOG directory path 
LOG_DIR : Path= Path(".logs/")

# Kaggle dataset reference
class KaggleDatasetRef:
    MAIN_DATASET = r"imsparsh/fma-free-music-archive-small-medium"
    MEDIUM_DATASET_FOLDER = r"fma_medium/fma_medium/"
    SMALL_DATASET_FOLDER = r"fma_small/fma_small/"
    METADATA_FOLDER = r"fma_metadata/"

# DATA directory path
FMA_SMALL_PATH = r"data/FMA_small/"

# TRASH directory path
TRASH_DIR : str = r".trash/"
HISTORIC_TRASH_CSV_NAME : str= "historic.csv"

DEFAULT_DATA_TO_SUPPRESS : List[str] = [
    "data/metadata/checksums",
    "data/metadata/raw_albums.csv",
    "data/metadata/raw_artists.csv",
    "data/metadata/raw_echonest.csv",
    "data/metadata/raw_genre.csv",
    "data/metadata/raw_tracks.csv"
]


# Aboute raw_genres.csv dataset
RAW_GENRES_PATH = r"data/metadata/genres.csv"
RAW_GENRES_PARENT_COL = 'parent'
MAIN_GENRE_CSV_PATH = r"data/metadata/main_genres.csv"

# About track.csv dataset
TRACK_PATH = r"data/metadata/tracks.csv"
FILTERED_TRACK_PATH = r"data/metadata/filtered_tracks.csv"
PATH_LABEL_CSV_PATH = r"data/metadata/path_labels.csv"

# About TFRecord output
TFRECORD_OUTPUT_DIR = r"data/tfrecords/"

#CLASS_NUMBER
NB_CLASSES : int = 6


#About audio processing
DEFAULT_SAMPLE_RATE : int = 22050
DEFAULT_SEGMENT_DURATION : float = 4.0  # in seconds
DEFAULT_OVERLAP : float = 0.5  # 50% overlap between segments

DEFAULT_N_FFT : int = 2048
DEFAULT_HOP_LENGTH : int = 512
DEFAULT_N_MELS : int = 128



# Dataset
TFRECORD_DIR = r"data/tfrecords"

# Audio / Spectrogram
N_MELS = 128
NUM_CHANNELS = 1

# Training
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

# Split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Paths
CHECKPOINT_DIR = "checkpoints"



# Enum pour splits plus reproductibles 
class Split:
    TRAIN = 0
    VAL = 1
    TEST = 2
