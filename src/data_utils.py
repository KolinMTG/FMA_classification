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

    # Ensure balanced classes
    min_count = df['label'].value_counts().min()
    df = df.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
    df = df[['path', 'label']]

    # Sauvegarde du CSV final
    csv_output_path = Path(csv_output_path)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
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
        csv_output_path=PATH_LABEL_CSV_PATH
    )


if __name__ == "__main__":
    build_csv_pipeline_00()


