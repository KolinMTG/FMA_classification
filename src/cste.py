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
FMA_SMALL_PATH = r"data/FMA_medium/fma_medium/"

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
TFRECORD_DIR = r"data/tfrecords"
DATA_SPLIT_CSV_PATH = r"data/metadata/dataset_split.csv"

#CLASS_NUMBER
NB_CLASSES : int = 6

#About audio processing
DEFAULT_SAMPLE_RATE : int = 22050
DEFAULT_SEGMENT_DURATION : float = 4.0  # in seconds
DEFAULT_OVERLAP : float = 0.5  # 50% overlap between segments

DEFAULT_N_FFT : int = 2048
DEFAULT_HOP_LENGTH : int = 512
DEFAULT_N_MELS : int = 128


# Audio / Spectrogram
N_MELS = 128
NUM_CHANNELS = 1


NUM_CLASSES = 6
# Training
class TrainingConstants:

    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3


# Paths
CHECKPOINT_DIR = "checkpoints"

# Data split ratios
class SplitRatios:
    TRAIN = 0.70
    VAL = 0.15
    TEST = 0.15

# Labels for one-hot encoding
class SplitLabels:
    TRAIN = 0
    VAL = 1
    TEST = 2

# Model defaults parameters
class ModelDefaults:
    NAME = "cnn_spectrogram"
    INPUT_SHAPE = (128, 173, 1)   # height, width, channels (ex: 4s audio spectrogram)
    OUTPUT_UNITS = 6              # nombre de classes
    CONV_LAYERS = [               # tuples (filters, kernel_size)
        (32, (3,3)),
        (64, (3,3)),
        (128, (3,3))
    ]
    CONV_ACTIVATIONS = ["relu", "relu", "relu"]
    POOL_SIZE = (2,2)
    DROPOUT_RATES = [0.3, 0.3, 0.4]  # après chaque block conv+pool
    DENSE_LAYERS = [128]          # fully connected après flatten
    DENSE_ACTIVATIONS = ["relu"]
    OPTIMIZER = "adam"
    LEARNING_RATE = 1e-3
    LOSS = "sparse_categorical_crossentropy"
    METRICS = ["accuracy"]



# Model CSV paths (3 different CSVs for different purposes)
class ModelsCSV:
    REGISTRY: str = "models/models_registry.csv"
    TRAINING: str = "models/model_training.csv"
    PERFORMANCE: str = "models/model_performance.csv"