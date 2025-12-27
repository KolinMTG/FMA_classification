from typing import List, Tuple, Dict, Optional
import tensorflow as tf
import numpy as np
import librosa
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import os
import json
from pathlib import Path
from collections import defaultdict

from src.cste import *
from src.logger import get_logger

log = get_logger("data_pretreat.log")

# ============================================================================
# AUDIO LOADING AND SEGMENTATION
# ============================================================================

def load_audio(audio_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file and return a mono waveform.
    """
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return waveform


def split_into_segments(waveform: np.ndarray, sample_rate: int,
                        segment_duration: float = DEFAULT_SEGMENT_DURATION, 
                        overlap: float = DEFAULT_OVERLAP) -> List[np.ndarray]:
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


# ============================================================================
# LOG-MEL SPECTROGRAM COMPUTATION
# ============================================================================

def compute_log_mel_spectrogram(
    segment: np.ndarray,
    sample_rate: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    min_db: float = -80.0,
) -> np.ndarray:
    """
    Compute a log-Mel spectrogram WITHOUT normalization.
    
    NOTE: We no longer normalize to [0, 1] here. Instead, we will compute
    dataset-level statistics (mean/std) from the training set and apply
    standardization (z-score normalization) consistently across all splits.
    
    This prevents information leakage from validation/test sets and provides
    more robust features for training.
    """
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel, ref=1.0)
    
    # Clip to fixed dynamic range (keeps extreme values in check)
    log_mel = np.clip(log_mel, min_db, 0.0)
    
    # Return raw log-Mel values without normalization
    return log_mel.astype(np.float32)


def apply_normalization(log_mel: np.ndarray, 
                       mean: np.ndarray, 
                       std: np.ndarray,
                       per_bin: bool = True) -> np.ndarray:
    """
    Apply z-score normalization (standardization) to log-Mel spectrogram.
    
    Args:
        log_mel: Log-Mel spectrogram of shape (n_mels, time_frames)
        mean: Mean values computed from training set
        std: Standard deviation values computed from training set
        per_bin: If True, mean/std are vectors of shape (n_mels,) for per-frequency normalization
                 If False, mean/std are scalars for global normalization
    
    Returns:
        Normalized log-Mel spectrogram with zero mean and unit variance
    """
    if per_bin:
        # Per-frequency bin normalization: mean and std have shape (n_mels, 1)
        # This accounts for different energy levels at different frequency bands
        return (log_mel - mean[:, np.newaxis]) / (std[:, np.newaxis] + 1e-8)
    else:
        # Global normalization: mean and std are scalars
        return (log_mel - mean) / (std + 1e-8)


def format_for_cnn(log_mel: np.ndarray) -> np.ndarray:
    """
    Add channel dimension to log-Mel spectrogram for CNN input.
    """
    return log_mel[..., np.newaxis]  # (H, W, 1)


# ============================================================================
# TFRECORD SERIALIZATION
# ============================================================================

def serialize_example(feature: np.ndarray, label: int) -> tf.train.Example:
    """
    Serialize a single CNN example for TFRecord.
    """
    height, width = feature.shape[:2]
    feature_bytes = feature.astype(np.float32).tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'spectrogram': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_bytes])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))
    }))
    return example


# ============================================================================
# STATISTICS COMPUTATION FOR NORMALIZATION
# ============================================================================

def compute_dataset_statistics(
    audio_paths: List[str],
    labels: List[int],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    per_bin: bool = True,
    max_samples: Optional[int] = None,
    # ========================================================================
    # MODIFICATION 1: Ajout des paramètres de spectrogramme configurables
    # Ces paramètres permettent de contrôler la résolution temps-fréquence
    # des spectrogrammes log-Mel. Ils DOIVENT être identiques à ceux utilisés
    # lors du traitement des données (train/val/test) pour garantir la cohérence.
    # ========================================================================
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    min_db: float = -80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation of log-Mel spectrograms across the dataset.
    
    This function processes all audio files in the provided list and accumulates
    statistics to compute dataset-level normalization parameters.
    
    Args:
        audio_paths: List of paths to audio files
        labels: List of corresponding labels (not used for statistics, but kept for consistency)
        sample_rate: Audio sample rate
        per_bin: If True, compute statistics per frequency bin (shape: n_mels)
                 If False, compute global statistics (scalars)
        max_samples: Optional limit on number of segments to use for statistics
                     (useful for large datasets to speed up computation)
        n_fft: FFT window size (affects frequency resolution)
        hop_length: Number of samples between successive frames (affects time resolution)
        n_mels: Number of Mel frequency bins (affects frequency granularity)
        min_db: Minimum decibel value for clipping (affects dynamic range)
    
    Returns:
        mean: Mean values (per-bin or global)
        std: Standard deviation values (per-bin or global)
    """
    log.info(f"Computing dataset statistics from {len(audio_paths)} audio files...")
    # ========================================================================
    # MODIFICATION 1bis: Log des paramètres de spectrogramme utilisés
    # Important pour la traçabilité et la reproductibilité des expériences
    # ========================================================================
    log.info(f"Spectrogram parameters: n_fft={n_fft}, hop_length={hop_length}, "
             f"n_mels={n_mels}, min_db={min_db}")
    
    all_spectrograms = []
    sample_count = 0
    
    for audio_path in tqdm(audio_paths, desc="Computing statistics"):
        if not os.path.exists(audio_path):
            log.warning(f"Audio file not found, skipping: {audio_path}")
            continue
            
        try:
            waveform = load_audio(audio_path, sample_rate)
            segments = split_into_segments(waveform, sample_rate)
            
            for segment in segments:
                # ============================================================
                # MODIFICATION 1ter: Passage des paramètres configurables
                # à compute_log_mel_spectrogram. Cela garantit que les stats
                # sont calculées avec les mêmes paramètres que les données.
                # ============================================================
                log_mel = compute_log_mel_spectrogram(
                    segment, 
                    sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    min_db=min_db
                )
                
                if log_mel.size == 0:
                    continue
                    
                all_spectrograms.append(log_mel)
                sample_count += 1
                
                # Early stopping if max_samples reached
                if max_samples and sample_count >= max_samples:
                    break
                    
        except Exception as e:
            log.warning(f"Error processing {audio_path} during statistics computation: {e}")
            continue
            
        if max_samples and sample_count >= max_samples:
            break
    
    if len(all_spectrograms) == 0:
        log.error("No valid spectrograms found for statistics computation")
        raise ValueError("Cannot compute statistics from empty dataset")
    
    # Stack all spectrograms: shape (num_samples, n_mels, time_frames)
    all_spectrograms = np.array(all_spectrograms)
    log.info(f"Collected {len(all_spectrograms)} spectrogram samples for statistics")
    
    if per_bin:
        # Compute mean and std per frequency bin across all time frames
        # Flatten time dimension: (num_samples * time_frames, n_mels)
        flattened = all_spectrograms.reshape(-1, all_spectrograms.shape[1])
        mean = np.mean(flattened, axis=0)  # shape: (n_mels,)
        std = np.std(flattened, axis=0)    # shape: (n_mels,)
        log.info(f"Computed per-bin statistics: mean shape {mean.shape}, std shape {std.shape}")
    else:
        # Compute global mean and std across all values
        mean = np.mean(all_spectrograms)
        std = np.std(all_spectrograms)
        log.info(f"Computed global statistics: mean={mean:.4f}, std={std:.4f}")
    
    return mean, std


def save_normalization_stats(mean: np.ndarray, std: np.ndarray, output_path: str,
                             # ============================================
                             # MODIFICATION 2: Sauvegarde des paramètres
                             # de spectrogramme avec les stats de normalisation.
                             # Cela permet de reconstruire exactement les mêmes
                             # spectrogrammes lors de l'inférence.
                             # ============================================
                             n_fft: int = DEFAULT_N_FFT,
                             hop_length: int = DEFAULT_HOP_LENGTH,
                             n_mels: int = DEFAULT_N_MELS,
                             min_db: float = -80.0):
    """
    Save normalization statistics AND spectrogram parameters to a JSON file.
    
    This ensures that during inference, we can reconstruct spectrograms with
    exactly the same parameters used during training, which is critical for
    model performance.
    """
    stats = {
        'mean': mean.tolist() if isinstance(mean, np.ndarray) else mean,
        'std': std.tolist() if isinstance(std, np.ndarray) else std,
        'per_bin': isinstance(mean, np.ndarray) and mean.ndim > 0,
        # ====================================================================
        # MODIFICATION 2bis: Ajout des paramètres de spectrogramme au JSON
        # Ces paramètres sont essentiels pour garantir la reproductibilité
        # ====================================================================
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'min_db': min_db,
        'sample_rate': DEFAULT_SAMPLE_RATE  # Ajout du sample_rate aussi
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    log.info(f"Normalization statistics saved to {output_path}")
    log.info(f"Saved spectrogram parameters: n_fft={n_fft}, hop_length={hop_length}, "
             f"n_mels={n_mels}, min_db={min_db}")


# ============================================================================
# STRATIFIED TRAIN/VAL/TEST SPLIT
# ============================================================================

def stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float = SplitRatios.TRAIN,
    val_ratio: float = SplitRatios.VAL,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate stratified train/val/test indices that maintain class balance.
    
    This ensures each split has approximately the same class distribution as
    the original dataset, which is crucial for fair evaluation.
    
    Args:
        labels: Array of integer labels
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_idx, val_idx, test_idx: Arrays of indices for each split
    """
    np.random.seed(seed)
    
    train_idx = []
    val_idx = []
    test_idx = []
    
    # Get unique classes
    unique_classes = np.unique(labels)
    log.info(f"Performing stratified split across {len(unique_classes)} classes")
    
    for cls in unique_classes:
        # Get all indices for this class
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        
        n_samples = len(cls_indices)
        n_train = int(train_ratio * n_samples)
        n_val = int(val_ratio * n_samples)
        
        # Split indices for this class
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:])
        
        log.info(f"Class {cls}: {n_train} train, {n_val} val, {n_samples - n_train - n_val} test")
    
    # Convert to numpy arrays and shuffle within each split
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    return train_idx, val_idx, test_idx


# ============================================================================
# PROCESSING FUNCTIONS WITH NORMALIZATION
# ============================================================================

def process_audio_to_tfrecord_with_norm(
    audio_path: str,
    label: int,
    tfrecord_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    per_bin: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    # ========================================================================
    # MODIFICATION 3: Ajout des paramètres de spectrogramme configurables
    # Ces paramètres DOIVENT être identiques à ceux utilisés pour calculer
    # les statistiques de normalisation (mean/std), sinon la normalisation
    # sera incorrecte et le modèle ne convergera pas correctement.
    # ========================================================================
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    min_db: float = -80.0,
) -> bool:
    """
    Process a single audio file with dataset-level normalization and write to TFRecord.
    
    Args:
        audio_path: Path to audio file
        label: Integer label for this audio
        tfrecord_path: Output TFRecord file path
        mean: Dataset mean for normalization
        std: Dataset standard deviation for normalization
        per_bin: Whether normalization is per-frequency-bin or global
        sample_rate: Audio sample rate
        n_fft: FFT window size (must match statistics computation)
        hop_length: Hop length between frames (must match statistics computation)
        n_mels: Number of Mel bins (must match statistics computation)
        min_db: Minimum dB for clipping (must match statistics computation)
    
    Returns:
        True if processed successfully, False if skipped
    """
    if not os.path.exists(audio_path):
        log.warning(f"Audio file not found, skipping: {audio_path}")
        return False

    try:
        waveform = load_audio(audio_path, sample_rate)
        segments = split_into_segments(waveform, sample_rate)

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for segment in segments:
                # ============================================================
                # MODIFICATION 3bis: Passage des paramètres configurables
                # à compute_log_mel_spectrogram. CRITIQUE: ces paramètres
                # doivent être identiques à ceux utilisés dans
                # compute_dataset_statistics() sinon les données seront
                # inconsistantes avec les statistiques de normalisation.
                # ============================================================
                log_mel = compute_log_mel_spectrogram(
                    segment, 
                    sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    min_db=min_db
                )
                
                if log_mel.size == 0:
                    log.warning(f"Empty spectrogram for {audio_path}, skipping segment")
                    continue
                
                # Apply dataset-level normalization
                normalized_log_mel = apply_normalization(log_mel, mean, std, per_bin)
                
                # Format for CNN input
                cnn_input = format_for_cnn(normalized_log_mel)
                
                # Serialize and write
                example = serialize_example(cnn_input, label)
                writer.write(example.SerializeToString())

        return True

    except Exception as e:
        log.warning(f"Error processing {audio_path}, skipping. Exception: {e}")
        return False


def worker_with_norm(args):
    """
    Wrapper for multiprocessing with normalization parameters.
    """
    return process_audio_to_tfrecord_with_norm(*args)


# ============================================================================
# MAIN PIPELINE WITH INTEGRATED SPLITTING AND NORMALIZATION
# ============================================================================

def build_data_pipeline_01(
    dataset: pd.DataFrame,
    output_dir: str = TFRECORD_OUTPUT_DIR,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_workers: int = NUM_WORKERS,
    train_ratio: float = SplitRatios.TRAIN,
    val_ratio: float = SplitRatios.VAL,
    per_bin_normalization: bool = True,
    seed: int = RANDOM_SEED,
    max_stats_samples: Optional[int] = None,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    min_db: float = -80.0,
) -> None:
    """
    Complete pipeline: split dataset, compute normalization stats, and build TFRecords.
    
    This function implements a robust preprocessing pipeline that:
    1. Performs stratified train/val/test split to maintain class balance
    2. Computes normalization statistics (mean/std) from TRAINING SET ONLY
    3. Applies consistent normalization to all splits (train/val/test)
    4. Writes TFRecords for each split with normalized spectrograms
    5. Saves a CSV file mapping TFRecord paths to labels and splits
    6. Saves normalization statistics for use during inference
    
    This approach prevents data leakage and overfitting that can occur with
    per-file normalization, where each audio file is normalized independently.
    
    Args:
        dataset: DataFrame with columns 'path' (audio file path) and 'label' (integer)
        output_dir: Directory to save TFRecord files and metadata
        sample_rate: Audio sampling rate
        num_workers: Number of parallel workers for processing
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        per_bin_normalization: If True, normalize per frequency bin; if False, global normalization
        seed: Random seed for reproducibility
        max_stats_samples: Optional limit on segments used for computing statistics
        n_fft: FFT window size for spectrogram computation
        hop_length: Hop length between frames for spectrogram computation
        n_mels: Number of Mel frequency bins for spectrogram computation
        min_db: Minimum dB value for clipping in spectrogram computation
    
    Directory structure created:
        output_dir/
            train/ - TFRecords for training set
            val/ - TFRecords for validation set
            test/ - TFRecords for test set
            dataset_split.csv - Mapping of TFRecord paths to labels and splits
            normalization_stats.json - Mean, std, and spectrogram parameters
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    log.info("="*80)
    log.info("STARTING AUDIO PREPROCESSING PIPELINE WITH DATASET-LEVEL NORMALIZATION")
    log.info("="*80)
    # ========================================================================
    # MODIFICATION 4bis: Log des paramètres de spectrogramme au début du pipeline
    # pour la traçabilité et le debugging
    # ========================================================================
    log.info(f"Spectrogram configuration:")
    log.info(f"  - n_fft: {n_fft}")
    log.info(f"  - hop_length: {hop_length}")
    log.info(f"  - n_mels: {n_mels}")
    log.info(f"  - min_db: {min_db}")
    log.info(f"  - sample_rate: {sample_rate}")
    
    # Step 1: Perform stratified split
    log.info("Step 1/4: Performing stratified train/val/test split...")
    labels = dataset['label'].values
    train_idx, val_idx, test_idx = stratified_split_indices(
        labels, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    
    log.info(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Step 2: Compute normalization statistics from TRAINING SET ONLY
    log.info("Step 2/4: Computing normalization statistics from TRAINING SET...")
    train_paths = dataset.iloc[train_idx]['path'].tolist()
    train_labels = dataset.iloc[train_idx]['label'].tolist()
    
    # ========================================================================
    # MODIFICATION 4ter: Passage des paramètres de spectrogramme à
    # compute_dataset_statistics. Ces paramètres définissent comment les
    # spectrogrammes sont calculés pour les statistiques de normalisation.
    # ========================================================================
    mean, std = compute_dataset_statistics(
        train_paths, 
        train_labels, 
        sample_rate=sample_rate,
        per_bin=per_bin_normalization, 
        max_samples=max_stats_samples,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        min_db=min_db
    )
    
    # Save normalization statistics
    stats_path = os.path.join(output_dir, "normalization_stats.json")
    # ========================================================================
    # MODIFICATION 4quater: Passage des paramètres de spectrogramme à
    # save_normalization_stats pour les sauvegarder dans le JSON.
    # Cela permet de les réutiliser lors de l'inférence.
    # ========================================================================
    save_normalization_stats(
        mean, 
        std, 
        stats_path,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        min_db=min_db
    )
    
    # Step 3: Process all splits with consistent normalization
    log.info("Step 3/4: Processing audio files and creating TFRecords...")
    
    split_info = [
        (train_idx, train_dir, SplitLabels.TRAIN, "train"),
        (val_idx, val_dir, SplitLabels.VAL, "validation"),
        (test_idx, test_dir, SplitLabels.TEST, "test")
    ]
    
    all_records = []
    num_workers = num_workers or mp.cpu_count()
    
    for indices, split_dir, split_label, split_name in split_info:
        log.info(f"Processing {split_name} set ({len(indices)} files)...")
        
        tasks = []
        for idx_pos, idx in enumerate(indices):
            row = dataset.iloc[idx]
            tfrecord_path = os.path.join(split_dir, f"{idx_pos:06d}.tfrecord")
            # ================================================================
            # MODIFICATION 4quinquies: Passage des paramètres de spectrogramme
            # à process_audio_to_tfrecord_with_norm pour CHAQUE split.
            # 
            # CRITIQUE: Les mêmes paramètres sont utilisés pour train, val ET test
            # pour garantir que tous les spectrogrammes sont calculés de manière
            # identique. Toute différence causerait des incohérences catastrophiques
            # dans le pipeline de normalisation.
            # ================================================================
            tasks.append((
                row['path'],
                int(row['label']),
                tfrecord_path,
                mean,
                std,
                per_bin_normalization,
                sample_rate,
                n_fft,           # Paramètre identique pour tous les splits
                hop_length,      # Paramètre identique pour tous les splits
                n_mels,          # Paramètre identique pour tous les splits
                min_db           # Paramètre identique pour tous les splits
            ))
            
            # Store record info for CSV
            all_records.append({
                'path': tfrecord_path,
                'label': int(row['label']),
                'split': split_label
            })
        
        # Process with multiprocessing
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.starmap(process_audio_to_tfrecord_with_norm, tasks),
                total=len(tasks),
                desc=f"Processing {split_name}"
            ))
        
        success_count = sum(results)
        failed_count = len(results) - success_count
        log.info(f"{split_name.capitalize()} set: {success_count} processed, {failed_count} skipped")
    
    # Step 4: Save dataset split CSV
    log.info("Step 4/4: Saving dataset split metadata...")
    df_split = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "dataset_split.csv")
    df_split.to_csv(csv_path, index=False)
    log.info(f"Dataset split CSV saved to {csv_path}")
    
    # Summary
    log.info("="*80)
    log.info("PIPELINE COMPLETED SUCCESSFULLY")
    log.info("="*80)
    log.info(f"Total samples: {len(all_records)}")
    log.info(f"Training samples: {len(train_idx)}")
    log.info(f"Validation samples: {len(val_idx)}")
    log.info(f"Test samples: {len(test_idx)}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Normalization stats: {stats_path}")
    log.info(f"Split metadata: {csv_path}")
    log.info("="*80)