import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

# Importer le logger depuis src
from src.logger import get_logger

# Logger global
logger = get_logger(log_file_name="inference_logger")


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(
    normalization_stats_path: str,
    label_mapping_path: str
) -> Dict:
    """
    Load all configuration from user-provided paths.
    
    Automatically extracts parameters from normalization_stats.json including:
    - Normalization statistics (mean, std, per_bin)
    - Audio parameters (sample_rate)
    - Spectrogram parameters (n_fft, hop_length, n_mels, min_db)
    - Label mapping from CSV
    
    Parameters
    ----------
    normalization_stats_path : str
        Path to normalization_stats.json file.
    label_mapping_path : str
        Path to label_mapping.csv file.
    
    Returns
    -------
    Dict
        Complete configuration dictionary.
    """
    logger.info("="*80)
    logger.info("LOADING CONFIGURATION")
    logger.info("="*80)
    
    # Load normalization statistics
    if not os.path.exists(normalization_stats_path):
        raise FileNotFoundError(
            f"normalization_stats.json not found: {normalization_stats_path}"
        )
    
    logger.info(f"Loading normalization stats: {normalization_stats_path}")
    with open(normalization_stats_path, 'r') as f:
        config = json.load(f)
    
    # Load label mapping
    if not os.path.exists(label_mapping_path):
        raise FileNotFoundError(
            f"label_mapping.csv not found: {label_mapping_path}"
        )
    
    logger.info(f"Loading label mapping: {label_mapping_path}")
    df = pd.read_csv(label_mapping_path)
    if 'genre' not in df.columns or 'label' not in df.columns:
        raise ValueError("label_mapping.csv must contain 'genre' and 'label' columns")
    
    config['label_mapping'] = dict(zip(df['label'], df['genre']))
    
    # Log configuration
    logger.info("\nLoaded Configuration:")
    logger.info(f"  Audio Parameters:")
    logger.info(f"    - Sample Rate: {config['sample_rate']} Hz")
    logger.info(f"  Spectrogram Parameters:")
    logger.info(f"    - n_fft: {config['n_fft']}")
    logger.info(f"    - hop_length: {config['hop_length']}")
    logger.info(f"    - n_mels: {config['n_mels']}")
    logger.info(f"    - min_db: {config['min_db']}")
    logger.info(f"  Normalization:")
    logger.info(f"    - Type: {'per-bin' if config['per_bin'] else 'global'}")
    logger.info(f"  Classes: {len(config['label_mapping'])} genres")
    
    return config


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def load_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    """Load an audio file and return a mono waveform."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    logger.info(f"Loading audio: {audio_path}")
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(waveform) / sample_rate
    
    logger.info(f"Audio loaded: {duration:.2f}s, {sr} Hz")
    return waveform


def split_into_segments(
    waveform: np.ndarray,
    sample_rate: int,
    segment_duration: float,
    overlap: float
) -> List[np.ndarray]:
    """Split waveform into overlapping fixed-length segments."""
    segment_length = int(segment_duration * sample_rate)
    hop_length = int(segment_length * (1 - overlap))
    
    segments = []
    for start in range(0, len(waveform) - segment_length + 1, hop_length):
        segments.append(waveform[start:start + segment_length])
    
    logger.info(f"Created {len(segments)} segments ({segment_duration}s each, {overlap*100:.0f}% overlap)")
    return segments


def compute_log_mel_spectrogram(
    segment: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    min_db: float
) -> np.ndarray:
    """Compute a log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=1.0)
    log_mel = np.clip(log_mel, min_db, 0.0)
    return log_mel.astype(np.float32)


def apply_normalization(
    log_mel: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    per_bin: bool
) -> np.ndarray:
    """Apply z-score normalization to log-Mel spectrogram."""
    if per_bin:
        mean = np.array(mean)
        std = np.array(std)
        if mean.ndim == 1:
            mean = mean[:, np.newaxis]
            std = std[:, np.newaxis]
        normalized = (log_mel - mean) / (std + 1e-8)
    else:
        normalized = (log_mel - mean) / (std + 1e-8)
    return normalized


def handle_spectrogram_shape(
    spectrogram: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """Handle spectrogram shape mismatches with padding/truncation."""
    current_shape = spectrogram.shape
    target_height, target_width = target_shape
    
    if current_shape[0] != target_height:
        raise ValueError(
            f"Frequency dimension mismatch: got {current_shape[0]}, "
            f"expected {target_height}. Check n_mels parameter."
        )
    
    current_width = current_shape[1]
    
    if current_width == target_width:
        return spectrogram
    elif current_width < target_width:
        pad_width = target_width - current_width
        return np.pad(spectrogram, pad_width=((0, 0), (0, pad_width)), mode='edge')
    else:
        return spectrogram[:, :target_width]


def process_segments(
    segments: List[np.ndarray],
    config: Dict,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """Process all segments into model-ready spectrograms."""
    logger.info("Computing spectrograms...")
    
    spectrograms = []
    for i, segment in enumerate(segments):
        # Compute log-Mel spectrogram
        log_mel = compute_log_mel_spectrogram(
            segment,
            config['sample_rate'],
            config['n_fft'],
            config['hop_length'],
            config['n_mels'],
            config['min_db']
        )
        
        # Apply normalization
        normalized = apply_normalization(
            log_mel,
            config['mean'],
            config['std'],
            config['per_bin']
        )
        
        # Handle shape
        reshaped = handle_spectrogram_shape(normalized, target_shape)
        
        # Add channel dimension
        formatted = reshaped[..., np.newaxis]
        spectrograms.append(formatted)
    
    batch = np.array(spectrograms)
    logger.info(f"Processed {len(spectrograms)} spectrograms, batch shape: {batch.shape}")
    return batch


# ============================================================================
# PREDICTION
# ============================================================================

def aggregate_predictions(
    predictions: np.ndarray,
    aggregation_method: str
) -> Tuple[int, np.ndarray, float]:
    """Aggregate predictions from multiple segments."""
    if aggregation_method == "soft_voting":
        aggregated_probs = np.mean(predictions, axis=0)
        predicted_class = np.argmax(aggregated_probs)
        confidence = float(aggregated_probs[predicted_class])
    
    elif aggregation_method == "hard_voting":
        segment_predictions = np.argmax(predictions, axis=1)
        unique, counts = np.unique(segment_predictions, return_counts=True)
        predicted_class = int(unique[np.argmax(counts)])
        aggregated_probs = np.mean(predictions, axis=0)
        confidence = float(aggregated_probs[predicted_class])
    
    elif aggregation_method == "max_prob":
        max_confidence_idx = np.argmax(np.max(predictions, axis=1))
        aggregated_probs = predictions[max_confidence_idx]
        predicted_class = np.argmax(aggregated_probs)
        confidence = float(aggregated_probs[predicted_class])
    
    else:
        raise ValueError(
            f"Invalid aggregation_method: {aggregation_method}. "
            "Must be 'soft_voting', 'hard_voting', or 'max_prob'."
        )
    
    logger.info(f"Aggregation ({aggregation_method}): class {predicted_class}, confidence {confidence:.4f}")
    return predicted_class, aggregated_probs, confidence


def predict_genre(
    model: tf.keras.Model,
    normalization_stats_path: str,
    label_mapping_path: str,
    audio_path: str,
    segment_duration: float = 3.0,
    overlap: float = 0.5,
    aggregation_method: str = "soft_voting",
    output_report_path: Optional[str] = None
) -> Dict:
    """
    Predict music genre from audio file using trained model.
    
    All audio and spectrogram parameters are automatically loaded from
    the normalization_stats.json file.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained TensorFlow/Keras model for genre classification.
    normalization_stats_path : str
        Path to normalization_stats.json file.
        All parameters (n_fft, hop_length, n_mels, min_db, sample_rate, etc.)
        are automatically loaded from this file.
    label_mapping_path : str
        Path to label_mapping.csv file (columns: genre, label).
    audio_path : str
        Path to audio file to classify (MP3, WAV, etc.).
    segment_duration : float, default=3.0
        Duration of each audio segment in seconds.
    overlap : float, default=0.5
        Overlap ratio between segments (0.0 to 1.0).
    aggregation_method : str, default='soft_voting'
        Method for aggregating segment predictions:
        - 'soft_voting': Average probabilities (recommended)
        - 'hard_voting': Majority vote of classes
        - 'max_prob': Use segment with highest confidence
    output_report_path : str, optional
        Path to save detailed prediction report.
    
    Returns
    -------
    Dict
        Detailed prediction results.
    
    Example
    -------
    >>> from src.logger import get_logger
    >>> logger = get_logger(log_file_name="inference_logger")
    >>> 
    >>> model = tf.keras.models.load_model('models/best_model.keras')
    >>> result = predict_genre(
    ...     model=model,
    ...     normalization_stats_path='data/normalization_stats.json',
    ...     label_mapping_path='data/label_mapping.csv',
    ...     audio_path='music/unknown_song.mp3'
    ... )
    >>> print(f"Predicted: {result['predicted_label']} ({result['confidence']:.2%})")
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info("="*80)
    logger.info("MUSIC GENRE PREDICTION")
    logger.info("="*80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Audio: {audio_path}")
    logger.info(f"Model: {model.name if hasattr(model, 'name') else 'unnamed'}")
    
    # Load configuration
    config = load_config(normalization_stats_path, label_mapping_path)
    
    # Get target shape from model
    model_input_shape = model.input_shape[1:]
    target_height, target_width = model_input_shape[0], model_input_shape[1]
    logger.info(f"\nModel expects input shape: {model_input_shape}")
    
    # Load and segment audio
    logger.info("\n" + "-"*80)
    waveform = load_audio(audio_path, config['sample_rate'])
    segments = split_into_segments(waveform, config['sample_rate'], segment_duration, overlap)
    
    if len(segments) == 0:
        raise RuntimeError(f"No segments created. Audio too short (min: {segment_duration}s)")
    
    # Process segments into spectrograms
    logger.info("-"*80)
    batch = process_segments(segments, config, (target_height, target_width))
    
    # Run prediction
    logger.info("-"*80)
    logger.info("Running model prediction...")
    predictions = model.predict(batch, verbose=0)
    
    # Aggregate results
    logger.info("-"*80)
    predicted_class, aggregated_probs, confidence = aggregate_predictions(
        predictions, aggregation_method
    )
    
    predicted_label = config['label_mapping'].get(predicted_class, f"Unknown ({predicted_class})")
    
    # Build results
    results = {
        'predicted_class': int(predicted_class),
        'predicted_label': predicted_label,
        'probabilities': aggregated_probs.tolist(),
        'confidence': float(confidence),
        'num_segments_analyzed': len(segments),
        'config_used': {
            'sample_rate': config['sample_rate'],
            'n_fft': config['n_fft'],
            'hop_length': config['hop_length'],
            'n_mels': config['n_mels'],
            'min_db': config['min_db'],
            'segment_duration': segment_duration,
            'overlap': overlap,
            'aggregation_method': aggregation_method
        },
        'timestamp': timestamp
    }
    
    # Log results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Predicted Genre: {predicted_label}")
    logger.info(f"Confidence: {confidence:.2%}")
    logger.info("\nProbability Distribution:")
    for label_idx in sorted(config['label_mapping'].keys()):
        genre_name = config['label_mapping'][label_idx]
        prob = aggregated_probs[label_idx]
        logger.info(f"  {genre_name:<20} {prob:.4f} ({prob*100:.2f}%)")
    
    # Save report if requested
    if output_report_path:
        save_prediction_report(results, config['label_mapping'], output_report_path)
    
    logger.info("="*80 + "\n")
    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def save_prediction_report(
    results: Dict,
    label_mapping: Dict[int, str],
    output_path: str
) -> None:
    """Save a detailed prediction report to a text file."""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MUSIC GENRE PREDICTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Prediction Date: {results['timestamp']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-"*80 + "\n")
            config = results['config_used']
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Number of Segments: {results['num_segments_analyzed']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PREDICTION RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Predicted Genre: {results['predicted_label']}\n")
            f.write(f"Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PROBABILITY DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Genre':<20} {'Probability':<12} {'Percentage':<12} {'Bar'}\n")
            f.write("-"*80 + "\n")
            
            probs = results['probabilities']
            sorted_indices = np.argsort(probs)[::-1]
            
            for idx in sorted_indices:
                genre = label_mapping.get(idx, f"Class {idx}")
                prob = probs[idx]
                bar = "█" * int(prob * 50)
                f.write(f"{genre:<20} {prob:<12.4f} {prob*100:<12.2f}% {bar}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Report saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_genre_batch(
    model: tf.keras.Model,
    normalization_stats_path: str,
    label_mapping_path: str,
    audio_paths: List[str],
    output_dir: str = "predictions",
    segment_duration: float = 3.0,
    overlap: float = 0.5,
    aggregation_method: str = "soft_voting"
) -> pd.DataFrame:
    """
    Predict genres for multiple audio files.
    
    Generates one prediction report per audio file automatically.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    normalization_stats_path : str
        Path to normalization_stats.json file.
    label_mapping_path : str
        Path to label_mapping.csv file.
    audio_paths : List[str]
        List of audio file paths to process.
    output_dir : str, default='predictions'
        Directory to save individual reports and summary.
    segment_duration : float, default=3.0
        Duration of each audio segment in seconds.
    overlap : float, default=0.5
        Overlap ratio between segments (0.0 to 1.0).
    aggregation_method : str, default='soft_voting'
        Method for aggregating segment predictions.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with prediction results for all files.
    
    Example
    -------
    >>> model = tf.keras.models.load_model('models/best_model.keras')
    >>> audio_files = ['song1.mp3', 'song2.mp3', 'song3.mp3']
    >>> results_df = predict_genre_batch(
    ...     model=model,
    ...     normalization_stats_path='data/normalization_stats.json',
    ...     label_mapping_path='data/label_mapping.csv',
    ...     audio_paths=audio_files,
    ...     output_dir='predictions'
    ... )
    """
    os.makedirs(output_dir, exist_ok=True)
    results_list = []
    
    for i, audio_path in enumerate(audio_paths, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(audio_paths)}: {audio_path}")
        logger.info(f"{'='*80}")
        
        try:
            # Generate report path automatically based on audio filename
            audio_name = Path(audio_path).stem
            report_path = os.path.join(output_dir, f"{audio_name}_report.txt")
            
            # Run prediction with explicit parameters (no **kwargs)
            result = predict_genre(
                model=model,
                normalization_stats_path=normalization_stats_path,
                label_mapping_path=label_mapping_path,
                audio_path=audio_path,
                segment_duration=segment_duration,
                overlap=overlap,
                aggregation_method=aggregation_method,
                output_report_path=report_path
            )
            
            results_list.append({
                'audio_path': audio_path,
                'audio_filename': Path(audio_path).name,
                'predicted_label': result['predicted_label'],
                'confidence': result['confidence'],
                'num_segments': result['num_segments_analyzed'],
                'report_path': report_path
            })
            
        except Exception as e:
            logger.error(f"ERROR processing {audio_path}: {e}")
            results_list.append({
                'audio_path': audio_path,
                'audio_filename': Path(audio_path).name,
                'predicted_label': 'ERROR',
                'confidence': 0.0,
                'num_segments': 0,
                'report_path': 'N/A'
            })
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results_list)
    summary_path = os.path.join(output_dir, "batch_predictions_summary.csv")
    df_results.to_csv(summary_path, index=False)
    logger.info(f"\nBatch summary saved: {summary_path}")
    logger.info(f"Total processed: {len(audio_paths)} files")
    logger.info(f"Successful: {(df_results['predicted_label'] != 'ERROR').sum()} files")
    logger.info(f"Failed: {(df_results['predicted_label'] == 'ERROR').sum()} files")
    
    return df_results


if __name__ == "__main__":
    # Example usage (requires a trained model and appropriate files)
    model_path = r"optimization_results/models/final_model.keras"
    normalization_stats_path = r"data/tfrecords_64/normalization_stats.json"
    label_mapping_path = r"data/metadata/label_mapping.csv"
    audio_files_list = [
        r"data/FMA_medium/fma_medium/000/000002.mp3",
        r"data/FMA_medium/fma_medium/000/000003.mp3",
        r"data/FMA_medium/fma_medium/000/000004.mp3",
        r"data/FMA_medium/fma_medium/000/000005.mp3"
    ]
    output_reports_path = [r"predictions/000002_report.txt", r"predictions/000003_report.txt", r"predictions/000004_report.txt", r"predictions/000005_report.txt"]
    
    # Load model
    model = tf.keras.models.load_model(model_path)

    result = predict_genre_batch(
        model=model,
        normalization_stats_path=normalization_stats_path,
        label_mapping_path=label_mapping_path,
        audio_paths=audio_files_list,
        output_dir=r"data/predictions",
        segment_duration=3.0,
        overlap=0.5,
        aggregation_method="soft_voting"
    )