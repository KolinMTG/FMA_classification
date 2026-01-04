import pandas as pd
import os
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from src.cste import *
from logger import get_logger
import pandas as pd
from pathlib import Path
import numpy as np
import json


log = get_logger("data_utils.log")


def id_to_folder_file_path(track_id: str) -> Tuple[str, str]:
    """Convert a track ID to its corresponding folder and file name in the FMA_SMALL dataset."""
    track_id_int = int(track_id)
    track_id_str = f"{track_id_int:06d}" 
    folder = str(track_id_str[:3])            
    file_name = track_id_str + ".mp3"
    return folder, file_name


def id_to_track_path(track_id: str, fma_small_path: str = FMA_SMALL_PATH) -> str:
    """Convert a track ID to its corresponding file path in the FMA_SMALL dataset.
    Args:
        track_id (str): The track ID to convert.
        fma_small_path (str): The base path to the FMA_SMALL dataset.
    Returns:
        str: The full file path to the track's MP3 file."""
    folder, file_name = id_to_folder_file_path(track_id)
    full_path = os.path.join(fma_small_path, folder, file_name)
    return full_path

def id_to_kaggle_file(track_id:str)-> str:
    """Convert a track ID to its corresponding file path in the Kaggle Medium dataset.
    Args:
        track_id (str): The track ID to convert.
    Returns:
        str: The full file path to the track's MP3 file in the Kaggle Medium dataset."""
    folder, file_name = id_to_folder_file_path(track_id)
    medium_dataset_folder = KaggleDatasetRef.MEDIUM_DATASET_FOLDER
    full_path = medium_dataset_folder + folder + "/" + file_name
    return full_path



def compute_input_shape(
    normalization_file: str,
    audio_duration: float,
    channels: int = 1,
    verbose: bool = False
) -> Tuple[int, int, int]:
    """
    Compute CNN model INPUT_SHAPE from a normalization statistics file.
    
    Args:
        normalization_file: Path to JSON statistics file
        audio_duration: Audio duration in seconds (all recordings must have same length)
        channels: Number of channels (1 for mono, 3 for RGB-like)
        verbose: Display computation information
    
    Returns:
        Tuple (height, width, channels) representing INPUT_SHAPE
    
    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys are missing
        ValueError: If values are invalid
    
    Example:
        >>> input_shape = compute_input_shape("stats.json", audio_duration=4.0)
        >>> print(input_shape)
        (128, 345, 1)
    """
    
    # 1. Load JSON file
    norm_path = Path(normalization_file)
    if not norm_path.exists():
        raise FileNotFoundError(f"File not found: {normalization_file}")
    
    with open(norm_path, 'r') as f:
        stats = json.load(f)
    
    # 2. Extract required parameters
    required_keys = ['n_mels', 'hop_length', 'sample_rate']
    missing_keys = [k for k in required_keys if k not in stats]
    if missing_keys:
        raise KeyError(f"Missing keys in JSON file: {missing_keys}")
    
    n_mels = stats['n_mels']
    hop_length = stats['hop_length']
    sample_rate = stats['sample_rate']
    
    # Validate values
    if n_mels <= 0 or hop_length <= 0 or sample_rate <= 0:
        raise ValueError(f"Invalid values: n_mels={n_mels}, "
                        f"hop_length={hop_length}, sample_rate={sample_rate}")
    
    if audio_duration <= 0:
        raise ValueError(f"Invalid audio duration: {audio_duration}")
    
    # 3. Compute height (frequency dimension)
    height = n_mels
    
    # 4. Compute width (temporal dimension)
    # Formula: n_frames = floor(n_samples / hop_length) + 1
    n_samples = int(sample_rate * audio_duration)
    width = (n_samples // hop_length) + 1
    
    # 5. Build shape
    input_shape = (height, width, channels)
    
    # 6. Display information if requested
    if verbose:
        print("="*60)
        print("INPUT_SHAPE COMPUTATION")
        print("="*60)
        print(f"Normalization file: {normalization_file}")
        print(f"\nExtracted parameters:")
        print(f"  - n_mels (mel bins)    : {n_mels}")
        print(f"  - hop_length           : {hop_length}")
        print(f"  - sample_rate          : {sample_rate} Hz")
        print(f"  - n_fft                : {stats.get('n_fft', 'N/A')}")
        print(f"\nAudio duration         : {audio_duration:.3f} seconds")
        print(f"Total samples          : {n_samples}")
        print(f"\nWidth computation:")
        print(f"  width = floor({n_samples} / {hop_length}) + 1")
        print(f"        = {width} frames")
        print(f"\nFinal INPUT_SHAPE      : {input_shape}")
        print(f"  - Height (frequency) : {height}")
        print(f"  - Width (time)       : {width}")
        print(f"  - Channels           : {channels}")
        print("="*60)
    
    return input_shape


def compute_duration_from_width(
    width: int,
    hop_length: int,
    sample_rate: int
) -> float:
    """
    Compute audio duration corresponding to a given spectrogram width.
    
    Args:
        width: Number of temporal frames
        hop_length: Hop between frames
        sample_rate: Sampling rate
    
    Returns:
        Duration in seconds
    
    Example:
        >>> duration = compute_duration_from_width(173, 256, 22050)
        >>> print(f"{duration:.2f}s")
        2.01s
    """
    n_samples = (width - 1) * hop_length
    duration = n_samples / sample_rate
    return duration


def suggest_optimal_shapes(
    normalization_file: str,
    target_durations: list = [1.0, 2.0, 3.0, 4.0, 5.0]
) -> dict:
    """
    Suggest multiple optimal INPUT_SHAPE for different audio durations.
    
    Args:
        normalization_file: Path to JSON file
        target_durations: List of durations to test
    
    Returns:
        Dictionary with suggested shapes
    
    Example:
        >>> suggestions = suggest_optimal_shapes("stats.json")
        >>> for duration, shape in suggestions.items():
        ...     print(f"{duration}s -> {shape}")
    """
    with open(normalization_file, 'r') as f:
        stats = json.load(f)
    
    n_mels = stats['n_mels']
    hop_length = stats['hop_length']
    sample_rate = stats['sample_rate']
    
    suggestions = {}
    
    print("="*70)
    print("OPTIMAL INPUT_SHAPE SUGGESTIONS")
    print("="*70)
    print(f"Parameters: n_mels={n_mels}, hop_length={hop_length}, sr={sample_rate}")
    print(f"\n{'Duration':<10} {'Shape':<20} {'Width':<10} {'Format':<15}")
    print("-"*70)
    
    for duration in target_durations:
        n_samples = int(sample_rate * duration)
        width = (n_samples // hop_length) + 1
        shape = (n_mels, width, 1)
        
        # Determine if it's square or close to square
        ratio = width / n_mels
        if abs(ratio - 1.0) < 0.1:
            format_type = "Square ✓"
        elif width > n_mels * 1.5:
            format_type = "Wide rectangle"
        elif width < n_mels * 0.7:
            format_type = "Tall rectangle"
        else:
            format_type = "Near-square"
        
        suggestions[duration] = shape
        print(f"{duration:.1f}s{'':<5} {str(shape):<20} {width:<10} {format_type:<15}")
    
    print("="*70)
    print("\nRecommendation: Prefer square or near-square formats")
    print("for better CNN performance with pooling layers.")
    print("="*70)
    
    return suggestions



def create_main_genres(raw_genres_path:str)->None:
    """Args : 
        raw_genres_path (str) : path to the raw main genres file
    Creates a raw main genres csv at the specified path (list of genres with no parent genres).
    """
    # Verify that the path exists and is a csv file
    if not os.path.exists(raw_genres_path):
        raise FileNotFoundError(f"The file {raw_genres_path} does not exist.")
    if not raw_genres_path.endswith('.csv'):
        raise ValueError("The file must be a CSV.")

    df = pd.read_csv(raw_genres_path)

    # Filter for main genres (no parent genres)
    main_genres_df = df[df[RAW_GENRES_PARENT_COL] == 0]
    main_genres_df.to_csv(MAIN_GENRE_CSV_PATH, index=False)

    log.info(f"Raw main genres file created at {MAIN_GENRE_CSV_PATH} with {len(main_genres_df)} entries.")


def extract_top_balanced_tracks(tracks_csv_path: str, nb_class: int,
                                output_csv_path: str, seed: int = 42) -> None:
    """
    Extract top nb_class genres, balance classes by sampling min_count,
    and save intermediate CSV with ['track_id', 'genre'].
    """
    tracks_csv = Path(tracks_csv_path)
    df = pd.read_csv(tracks_csv, header=[0,1,2])
    # Flatten MultiIndex columns
    df.columns = ['_'.join([str(c) for c in col if str(c) != 'nan']).strip('_') for col in df.columns]
    track_id_col = next(c for c in df.columns if 'track_id' in c.lower())
    genre_col = next(c for c in df.columns if 'genre_top' in c.lower())

    genre_counts = df[genre_col].value_counts()
    top_genres = genre_counts.nlargest(nb_class).index.tolist()
    min_count = genre_counts[top_genres].min()
    log.info(f"Top {nb_class} genres: {top_genres}, sampling {min_count} per genre")

    balanced_rows = []
    for genre in top_genres:
        genre_df = df[df[genre_col] == genre][[track_id_col, genre_col]].copy()
        if len(genre_df) > min_count:
            genre_df = genre_df.sample(n=min_count, random_state=seed)
        balanced_rows.append(genre_df)
    balanced_df = pd.concat(balanced_rows, ignore_index=True)
    balanced_df.columns = ['track_id', 'genre']
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_csv_path, index=False)
    log.info(f"Intermediate filtered CSV saved: {output_csv_path} ({len(balanced_df)} entries)")

def ensure_existance(track_id:str,fma_small_path:str = FMA_SMALL_PATH)-> bool:
    """Check if the audio file for the given track ID exists locally.
    """
    local_path = id_to_track_path(track_id, fma_small_path) # Ex : data/FMA_small/000/000000.mp3
    if os.path.exists(local_path):
        return True
    return False


def build_path_label_csv(csv_input_path: str, csv_output_path: str,
                         fma_small_path: str = FMA_SMALL_PATH) -> None:
    """
    Convert 'track_id' to full path, verify files exist (or download if missing), 
    encode genre as label, skip missing files, and ensure balanced classes.
    
    Also generates a label_mapping.csv file with the genre-to-label mapping.
    """
    df = pd.read_csv(csv_input_path)

    # Vérification existence ou téléchargement
    def check_or_download(tid):
        return ensure_existance(track_id=tid, fma_small_path=fma_small_path)

    exists_mask = df['track_id'].apply(check_or_download)
    missing_count = (~exists_mask).sum()
    if missing_count > 0:
        log.warning(f"{missing_count} files missing and will be skipped")
    df = df[exists_mask].copy()

    # Ajouter le chemin complet
    df['path'] = df['track_id'].apply(lambda tid: id_to_track_path(tid, fma_small_path))

    # Encode genres as integers
    unique_genres = sorted(df['genre'].unique())
    genre2id = {g: i for i, g in enumerate(unique_genres)}
    df['label'] = df['genre'].map(genre2id)

    # ========================================================================
    # NOUVEAU : Créer et sauvegarder le fichier de mapping genre/label
    # ========================================================================
    mapping_df = pd.DataFrame({
        'genre': list(genre2id.keys()),
        'label': list(genre2id.values())
    })
    
    # Sauvegarder dans le même dossier que le CSV de sortie
    mapping_output_path = MAPPING_GENRE_LABEL_CSV_PATH
    mapping_df.to_csv(mapping_output_path, index=False)
    log.info(f"Label mapping saved: {mapping_output_path}")
    log.info(f"Mapping: {dict(zip(mapping_df['genre'], mapping_df['label']))}")
    # ========================================================================

    # Ensure balanced classes
    min_count = df['label'].value_counts().min()
    df = df.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
    df = df[['path', 'label']]

    # Save final CSV
    df.to_csv(csv_output_path, index=False)
    log.info(f"Final path/label CSV saved: {csv_output_path} ({len(df)} entries, {min_count} per class)")



def build_csv_pipeline_00():
    """Run the full CSV building pipeline."""
    create_main_genres(RAW_GENRES_PATH)
    extract_top_balanced_tracks(
        tracks_csv_path=TRACK_PATH,
        nb_class=NB_CLASSES,
        output_csv_path=FILTERED_TRACK_PATH
    )

    build_path_label_csv(
        csv_input_path=FILTERED_TRACK_PATH,
        csv_output_path=PATH_LABEL_CSV_PATH,
    )


if __name__ == "__main__":
    build_csv_pipeline_00()


