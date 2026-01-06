import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

from src.logger import get_logger

logger = get_logger(log_file_name="inference_logger")


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(normalization_stats_path: str, label_mapping_path: str) -> Dict:
    """Load normalization stats and label mapping from files."""
    
    if not os.path.exists(normalization_stats_path):
        raise FileNotFoundError(f"Normalization stats not found: {normalization_stats_path}")
    
    with open(normalization_stats_path, 'r') as f:
        config = json.load(f)
    
    if not os.path.exists(label_mapping_path):
        raise FileNotFoundError(f"Label mapping not found: {label_mapping_path}")
    
    df = pd.read_csv(label_mapping_path)
    if 'genre' not in df.columns or 'label' not in df.columns:
        raise ValueError("label_mapping.csv must contain 'genre' and 'label' columns")
    
    config['label_mapping'] = dict(zip(df['label'], df['genre']))
    
    logger.info(f"Configuration loaded: {len(config['label_mapping'])} genres, "
                f"SR={config['sample_rate']}Hz, n_mels={config['n_mels']}")
    
    return config


# ============================================================================
# TFRECORD PROCESSING
# ============================================================================

def parse_tfrecord_example(example_proto):
    """
    Parse TFRecord example for inference.
    Only uses spectrogram + spatial dimensions.
    """

    features = {
        "spectrogram": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        # label exists in TFRecord but is intentionally ignored
    }

    parsed = tf.io.parse_single_example(example_proto, features)

    # Decode raw bytes as float32
    spectrogram = tf.io.decode_raw(parsed["spectrogram"], tf.float32)

    # Restore original shape (mono-channel)
    h = parsed["height"]
    w = parsed["width"]

    spectrogram = tf.reshape(spectrogram, (h, w, 1))

    return spectrogram


def load_tfrecord_segments(tfrecord_path: str, model_input_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Load all spectrogram segments from a TFRecord file.
    
    Returns only the spectrograms - labels are ignored as this is inference.
    """
    
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    spectrograms = []
    
    target_height, target_width, target_channels = model_input_shape
    
    for raw_record in dataset:
        spectrogram = parse_tfrecord_example(raw_record)
        
        # Reshape to expected model input
        spectrogram = tf.reshape(spectrogram, [target_height, target_width, target_channels])
        spectrograms.append(spectrogram.numpy())
    
    if len(spectrograms) == 0:
        raise RuntimeError(f"No segments found in TFRecord: {tfrecord_path}")
    
    return np.array(spectrograms)


def predict_from_tfrecord(
    model: tf.keras.Model,
    label_mapping: Dict[int, str],
    tfrecord_path: str,
    aggregation_method: str = "soft_voting"
) -> Dict:
    """
    Predict genre from preprocessed TFRecord file.
    
    The TFRecord label is completely ignored - prediction is made by the model.
    """
    
    model_input_shape = model.input_shape[1:]
    batch = load_tfrecord_segments(tfrecord_path, model_input_shape)
    
    predictions = model.predict(batch, verbose=0)
    predicted_class, aggregated_probs, confidence = aggregate_predictions(predictions, aggregation_method)
    
    # Get top-N predictions sorted by probability
    top_n_predictions = get_top_n_predictions(aggregated_probs, label_mapping)
    
    predicted_label = label_mapping.get(predicted_class, f"Unknown ({predicted_class})")
    
    return {
        'predicted_class': int(predicted_class),
        'predicted_label': predicted_label,
        'probabilities': aggregated_probs.tolist(),
        'confidence': float(confidence),
        'num_segments': len(batch),
        'source': 'tfrecord',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'top_n_predictions': top_n_predictions
    }


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def load_audio(audio_path: str, sample_rate: int) -> np.ndarray:
    """Load audio file and return mono waveform."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return waveform


def split_into_segments(waveform: np.ndarray, sample_rate: int, segment_duration: float, overlap: float) -> List[np.ndarray]:
    """Split waveform into overlapping segments."""
    segment_length = int(segment_duration * sample_rate)
    hop_length = int(segment_length * (1 - overlap))
    
    segments = []
    for start in range(0, len(waveform) - segment_length + 1, hop_length):
        segments.append(waveform[start:start + segment_length])
    
    return segments


def compute_log_mel_spectrogram(segment: np.ndarray, sample_rate: int, n_fft: int, hop_length: int, n_mels: int, min_db: float) -> np.ndarray:
    """Compute log-Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=segment, sr=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=1.0)
    log_mel = np.clip(log_mel, min_db, 0.0)
    return log_mel.astype(np.float32)


def apply_normalization(log_mel: np.ndarray, mean: np.ndarray, std: np.ndarray, per_bin: bool) -> np.ndarray:
    """Apply z-score normalization."""
    if per_bin:
        mean = np.array(mean)
        std = np.array(std)
        if mean.ndim == 1:
            mean = mean[:, np.newaxis]
            std = std[:, np.newaxis]
    
    return (log_mel - mean) / (std + 1e-8)


def handle_spectrogram_shape(spectrogram: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Handle shape mismatches with padding/truncation."""
    target_height, target_width = target_shape
    current_height, current_width = spectrogram.shape
    
    if current_height != target_height:
        raise ValueError(f"Frequency dimension mismatch: got {current_height}, expected {target_height}")
    
    if current_width < target_width:
        pad_width = target_width - current_width
        return np.pad(spectrogram, pad_width=((0, 0), (0, pad_width)), mode='edge')
    elif current_width > target_width:
        return spectrogram[:, :target_width]
    
    return spectrogram


def process_segments(segments: List[np.ndarray], config: Dict, target_shape: Tuple[int, int]) -> np.ndarray:
    """Process audio segments into model-ready spectrograms."""
    
    spectrograms = []
    for segment in segments:
        log_mel = compute_log_mel_spectrogram(
            segment, config['sample_rate'], config['n_fft'],
            config['hop_length'], config['n_mels'], config['min_db']
        )
        
        normalized = apply_normalization(log_mel, config['mean'], config['std'], config['per_bin'])
        reshaped = handle_spectrogram_shape(normalized, target_shape)
        spectrograms.append(reshaped[..., np.newaxis])
    
    return np.array(spectrograms)


def predict_from_audio(
    model: tf.keras.Model,
    config: Dict,
    audio_path: str,
    segment_duration: float = 3.0,
    overlap: float = 0.5,
    aggregation_method: str = "soft_voting"
) -> Dict:
    """Predict genre from raw audio file."""
    
    model_input_shape = model.input_shape[1:]
    target_height, target_width = model_input_shape[0], model_input_shape[1]
    
    waveform = load_audio(audio_path, config['sample_rate'])
    segments = split_into_segments(waveform, config['sample_rate'], segment_duration, overlap)
    
    if len(segments) == 0:
        raise RuntimeError(f"Audio too short (min: {segment_duration}s)")
    
    batch = process_segments(segments, config, (target_height, target_width))
    predictions = model.predict(batch, verbose=0)
    predicted_class, aggregated_probs, confidence = aggregate_predictions(predictions, aggregation_method)
    
    # Get top-N predictions sorted by probability
    top_n_predictions = get_top_n_predictions(aggregated_probs, config['label_mapping'])
    
    predicted_label = config['label_mapping'].get(predicted_class, f"Unknown ({predicted_class})")
    
    return {
        'predicted_class': int(predicted_class),
        'predicted_label': predicted_label,
        'probabilities': aggregated_probs.tolist(),
        'confidence': float(confidence),
        'num_segments': len(segments),
        'source': 'audio',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'top_n_predictions': top_n_predictions
    }


# ============================================================================
# PREDICTION AGGREGATION
# ============================================================================

def aggregate_predictions(predictions: np.ndarray, method: str) -> Tuple[int, np.ndarray, float]:
    """Aggregate predictions from multiple segments."""
    
    if method == "soft_voting":
        aggregated_probs = np.mean(predictions, axis=0)
        predicted_class = np.argmax(aggregated_probs)
        confidence = float(aggregated_probs[predicted_class])
    
    elif method == "hard_voting":
        segment_predictions = np.argmax(predictions, axis=1)
        unique, counts = np.unique(segment_predictions, return_counts=True)
        predicted_class = int(unique[np.argmax(counts)])
        aggregated_probs = np.mean(predictions, axis=0)
        confidence = float(aggregated_probs[predicted_class])
    
    elif method == "max_prob":
        max_confidence_idx = np.argmax(np.max(predictions, axis=1))
        aggregated_probs = predictions[max_confidence_idx]
        predicted_class = np.argmax(aggregated_probs)
        confidence = float(aggregated_probs[predicted_class])
    
    else:
        raise ValueError(f"Invalid aggregation_method: {method}")
    
    return predicted_class, aggregated_probs, confidence


def get_top_n_predictions(probabilities: np.ndarray, label_mapping: Dict[int, str]) -> List[Dict[str, any]]:
    """
    Get all class predictions sorted by probability (descending).
    
    Args:
        probabilities: Array of class probabilities
        label_mapping: Dictionary mapping class indices to genre names
    
    Returns:
        List of dictionaries with 'class_idx', 'label', and 'confidence' for each class
    """
    # Get indices sorted by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    
    top_n = []
    for idx in sorted_indices:
        top_n.append({
            'class_idx': int(idx),
            'label': label_mapping.get(idx, f"Unknown ({idx})"),
            'confidence': float(probabilities[idx])
        })
    
    return top_n


# ============================================================================
# REPORT GENERATION
# ============================================================================

def save_prediction_report(results: Dict, label_mapping: Dict[int, str], output_path: str) -> None:
    """Save detailed prediction report to text file."""
    
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MUSIC GENRE PREDICTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Prediction Date: {results['timestamp']}\n")
            f.write(f"Source: {results.get('source', 'N/A')}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("PREDICTION RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Predicted Genre: {results['predicted_label']}\n")
            f.write(f"Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)\n")
            f.write(f"Segments Analyzed: {results['num_segments']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("PROBABILITY DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Genre':<20} {'Probability':<12} {'Bar'}\n")
            f.write("-" * 80 + "\n")
            
            probs = results['probabilities']
            sorted_indices = np.argsort(probs)[::-1]
            
            for idx in sorted_indices:
                genre = label_mapping.get(idx, f"Class {idx}")
                prob = probs[idx]
                bar = "█" * int(prob * 50)
                f.write(f"{genre:<20} {prob:<12.4f} {bar}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_single_file(
    model: tf.keras.Model,
    file_path: str,
    label_mapping: Dict[int, str],
    config: Optional[Dict] = None,
    segment_duration: float = 3.0,
    overlap: float = 0.5,
    aggregation_method: str = "soft_voting"
) -> Dict:
    """Predict genre for a single file (auto-detects type)."""
    
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == '.tfrecord':
            result = predict_from_tfrecord(model, label_mapping, file_path, aggregation_method)
        elif file_ext in ['.mp3', '.wav', '.flac', '.ogg']:
            if config is None:
                raise ValueError("config required for audio file processing")
            result = predict_from_audio(model, config, file_path, segment_duration, overlap, aggregation_method)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        result['file_path'] = file_path
        result['filename'] = Path(file_path).name
        result['file_type'] = 'tfrecord' if file_ext == '.tfrecord' else 'audio'
        result['status'] = 'success'
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return {
            'file_path': file_path,
            'filename': Path(file_path).name,
            'file_type': 'unknown',
            'predicted_label': 'ERROR',
            'confidence': 0.0,
            'num_segments': 0,
            'status': 'error',
            'error_message': str(e),
            'top_n_predictions': []
        }


def predict_folder(
    model: tf.keras.Model,
    input_folder: str,
    output_folder: str,
    label_mapping_path: str,
    normalization_stats_path: Optional[str] = None,
    segment_duration: float = 3.0,
    overlap: float = 0.5,
    aggregation_method: str = "soft_voting",
    audio_extensions: Tuple[str, ...] = ('.mp3', '.wav', '.flac', '.ogg'),
    tfrecord_extensions: Tuple[str, ...] = ('.tfrecord',)
) -> pd.DataFrame:
    """
    Predict genres for all files in a folder.
    
    Automatically detects file types:
    - Audio files: requires normalization_stats_path
    - TFRecord files: preprocessed, no normalization needed
    
    Returns DataFrame with prediction results for all files, including top-N predictions.
    """
    
    logger.info("=" * 80)
    logger.info(f"FOLDER PREDICTION: {input_folder}")
    logger.info("=" * 80)
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Scan for files
    audio_files = []
    tfrecord_files = []
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(audio_extensions):
                audio_files.append(file_path)
            elif file.lower().endswith(tfrecord_extensions):
                tfrecord_files.append(file_path)
    
    all_files = audio_files + tfrecord_files
    total_files = len(all_files)
    
    logger.info(f"Found {len(audio_files)} audio files, {len(tfrecord_files)} TFRecord files")
    
    # Validate configuration
    if len(audio_files) > 0 and normalization_stats_path is None:
        raise ValueError("normalization_stats_path required for audio file processing")
    
    # Load configuration if needed
    config = None
    label_mapping = None
    
    if len(audio_files) > 0:
        config = load_config(normalization_stats_path, label_mapping_path)
        label_mapping = config['label_mapping']
    else:
        df = pd.read_csv(label_mapping_path)
        label_mapping = dict(zip(df['label'], df['genre']))
    
    # Determine number of classes for column generation
    num_classes = len(label_mapping)
    
    # Process all files
    all_results = []
    
    for i, file_path in enumerate(all_files, 1):
        logger.info(f"Processing file {i}/{total_files}: {Path(file_path).name}")
        
        result = predict_single_file(
            model, file_path, label_mapping, config,
            segment_duration, overlap, aggregation_method
        )
        
        # Save individual report if successful
        if result['status'] == 'success':
            file_name = Path(file_path).stem
            report_path = os.path.join(output_folder, f"{file_name}_report.txt")
            save_prediction_report(result, label_mapping, report_path)
            result['report_path'] = report_path
        else:
            result['report_path'] = 'N/A'
        
        # Format result for DataFrame with top-N predictions
        row_data = {
            'file_path': result['file_path'],
            'filename': result['filename'],
            'file_type': result.get('file_type', 'unknown'),
            'predicted_label': result.get('predicted_label', 'ERROR'),
            'confidence': result.get('confidence', 0.0),
            'num_segments': result.get('num_segments', 0),
            'status': result['status'],
            'report_path': result.get('report_path', 'N/A')
        }
        
        # Add top-N predictions as separate columns
        top_n_predictions = result.get('top_n_predictions', [])
        for rank, pred_data in enumerate(top_n_predictions, start=1):
            row_data[f'predicted_label_{rank}'] = pred_data['label']
            row_data[f'confidence_{rank}'] = pred_data['confidence']
        
        # Fill remaining columns with None if fewer than num_classes predictions
        for rank in range(len(top_n_predictions) + 1, num_classes + 1):
            row_data[f'predicted_label_{rank}'] = None
            row_data[f'confidence_{rank}'] = None
        
        all_results.append(row_data)
    
    # Create summary DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Reorder columns to have base columns first, then top-N predictions in order
    base_columns = ['file_path', 'filename', 'file_type', 'predicted_label', 
                    'confidence', 'num_segments', 'status', 'report_path']
    
    # Generate ordered top-N columns
    topn_columns = []
    for rank in range(1, num_classes + 1):
        topn_columns.append(f'predicted_label_{rank}')
        topn_columns.append(f'confidence_{rank}')
    
    # Reorder DataFrame columns
    ordered_columns = base_columns + topn_columns
    df_results = df_results[ordered_columns]
    
    summary_path = os.path.join(output_folder, ".predictions_summary.csv")
    df_results.to_csv(summary_path, index=False)
    
    # Summary statistics
    successful = (df_results['status'] == 'success').sum()
    failed = (df_results['status'] == 'error').sum()
    
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Summary saved: {summary_path}")
    logger.info("=" * 80)
    
    return df_results


# ============================================================================
# STANDALONE FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

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
    """Predict genre from audio file (backward compatibility wrapper)."""
    
    config = load_config(normalization_stats_path, label_mapping_path)
    result = predict_from_audio(model, config, audio_path, segment_duration, overlap, aggregation_method)
    
    if output_report_path:
        save_prediction_report(result, config['label_mapping'], output_report_path)
    
    return result


if __name__ == "__main__":
    # Example usage
    model_path = "optimization_results/models/final_model.keras"
    normalization_stats_path = "data/tfrecords_64/normalization_stats.json"
    label_mapping_path = "data/metadata/label_mapping.csv"
    input_folder = r"data/tfrecords_64/test/"
    output_folder = r"predictions/test_tfrecords/"

    
    model = tf.keras.models.load_model(model_path)
    results_df = predict_folder(
        model=model,
        input_folder=input_folder,
        output_folder=output_folder,
        label_mapping_path=label_mapping_path,
        normalization_stats_path=normalization_stats_path
    )
    print(results_df.head())