from typing import List, Tuple, Dict
import tensorflow as tf
import numpy as np
import librosa
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import os

from src.cste import *
from src.logger import get_logger

log = get_logger("data_pretreat.log")

def load_audio(audio_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file and return a mono waveform.
    """
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return waveform


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """
    Peak normalization of audio waveform.
    """
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    else :
        log.warning("Max value of waveform is zero, returning original waveform.")
    return waveform

def split_into_segments(waveform: np.ndarray, sample_rate: int,
                        segment_duration: float = DEFAULT_SEGMENT_DURATION, overlap: float = DEFAULT_OVERLAP) -> List[np.ndarray]:
    """
    Split waveform into overlapping fixed-length segments.
    !ATTENTION: overlap should be between 0 and 1
    """
    segment_length = int(segment_duration * sample_rate)
    hop_length = int(segment_length * (1 - overlap))
    segments = []
    for start in range(0, len(waveform) - segment_length + 1, hop_length):
        segments.append(waveform[start:start+segment_length])
    return segments

def compute_log_mel_spectrogram(segment: np.ndarray, sample_rate: int,
                                n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH, n_mels: int = DEFAULT_N_MELS) -> np.ndarray:
    """
    Compute log-Mel spectrogram from audio segment.
    """
    mel = librosa.feature.melspectrogram(y=segment, sr=sample_rate, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def format_for_cnn(log_mel: np.ndarray) -> np.ndarray:
    """
    Add channel dimension to log-Mel spectrogram for CNN input.
    """
    return log_mel[..., np.newaxis]  # (H, W, 1)

def serialize_example(feature: np.ndarray, label: int) -> tf.train.Example:
    """
    Serialize a single CNN example for TFRecord.
    """
    height, width = feature.shape[:2]  # save shape first
    feature_bytes = feature.astype(np.float32).tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'spectrogram': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))
    }))
    return example

def process_audio_to_tfrecord(audio_path: str, label: int, tfrecord_path: str,
                              sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
    """
    Process a single audio file and append segments to TFRecord.
    If audio cannot be loaded, logs a warning and skips.

    Returns:
        True if processed successfully, False if skipped.
    """
    if not os.path.exists(audio_path):
        log.warning(f"Audio file not found, skipping: {audio_path}")
        return False

    try:
        # log.info(f"Processing {audio_path}")
        waveform = load_audio(audio_path, sample_rate)
        waveform = normalize_audio(waveform)
        segments = split_into_segments(waveform, sample_rate)

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for segment in segments:
                log_mel = compute_log_mel_spectrogram(segment, sample_rate)
                cnn_input = format_for_cnn(log_mel)

                if log_mel.size == 0:
                    log.warning(f"Empty spectrogram for {audio_path}, skipping")
                    continue

                example = serialize_example(cnn_input, label)
                writer.write(example.SerializeToString())

        return True

    except Exception as e:
        log.warning(f"Error processing {audio_path}, skipping. Exception: {e}")
        return False




# --- worker doit être global ---
def worker(args):
    """
    Wrapper for processing a single audio to TFRecord.
    Used for multiprocessing pool.
    """
    process_audio_to_tfrecord(*args)


def build_tfrecord_from_dataframe_01(
    dataset: pd.DataFrame,
    output_dir: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_workers: int = NUM_WORKERS
):
    """
    Build TFRecord dataset from a DataFrame containing 'path' and 'label' columns.

    Args:
        dataset : pd.DataFrame
            DataFrame with columns:
                - 'path': path to .mp3 file
                - 'label': integer label
        output_dir : str
            Directory to save TFRecord files.
        sample_rate : int
            Sampling rate for audio.
        num_workers : int
            Number of CPU processes for multiprocessing.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    num_workers = num_workers or mp.cpu_count()

    # Prepare tasks
    tasks = []
    for idx, row in dataset.iterrows():
        tfrecord_path = os.path.join(output_dir, f"{idx:06d}.tfrecord")
        tasks.append((row['path'], int(row['label']), tfrecord_path, sample_rate))

    log.info(f"Starting TFRecord preprocessing with {num_workers} workers on {len(tasks)} files")

    # Multiprocessing
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.starmap(process_audio_to_tfrecord, tasks),
                            total=len(tasks), desc="Processing audio files"))

    success_count = sum(results)
    failed_count = len(results) - success_count
    log.info("------------------------------------------------------------------------------------------")
    log.info(f"TFRecord dataset build completed: {success_count} files processed, {failed_count} skipped")
    log.info("------------------------------------------------------------------------------------------")
    log.info("TFRecord dataset build completed")





