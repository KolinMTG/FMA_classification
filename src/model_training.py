import os
from pathlib import Path
import tensorflow as tf
from typing import List, Tuple, Optional
from datetime import datetime
import pandas as pd
from src.cste import *
from src.logger import get_logger

log = get_logger("model_training")

# ============================================================================
# TFRECORD PARSING
# ============================================================================

def parse_tfrecord(example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse a TFRecord example into (spectrogram, label) for training.
    
    The spectrogram is already normalized during preprocessing using dataset-level
    statistics, so no additional normalization is applied here.

    Parameters
    ----------
    example_proto : tf.Tensor
        Serialized TFRecord example.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        spectrogram : Tensor of shape (height, width, channels)
        label : integer Tensor (for sparse categorical crossentropy)
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


# ============================================================================
# DATASET BUILDING FROM NEW STRUCTURE
# ============================================================================

def get_tfrecord_paths_from_split(
    tfrecord_dir: str,
    split_name: str
) -> List[str]:
    """
    Get all TFRecord file paths from a specific split subdirectory.
    
    With the new data structure, TFRecords are organized in subdirectories:
    - tfrecord_dir/train/*.tfrecord
    - tfrecord_dir/val/*.tfrecord
    - tfrecord_dir/test/*.tfrecord
    
    Parameters
    ----------
    tfrecord_dir : str
        Base directory containing train/val/test subdirectories.
    split_name : str
        Name of the split subdirectory ('train', 'val', or 'test').
    
    Returns
    -------
    List[str]
        List of paths to all TFRecord files in the split directory.
    """
    split_dir = Path(tfrecord_dir) / split_name
    
    if not split_dir.exists():
        log.warning(f"Split directory not found: {split_dir}")
        return []
    
    # Get all .tfrecord files sorted by name
    tfrecord_paths = sorted(split_dir.glob("*.tfrecord"))
    tfrecord_paths = [str(p) for p in tfrecord_paths]
    
    log.info(f"Found {len(tfrecord_paths)} TFRecord files in {split_name} split")
    
    return tfrecord_paths


def build_dataset_from_tfrecords(
    tfrecord_paths: List[str],
    batch_size: int,
    shuffle: bool = True,
    shuffle_buffer_size: int = 2000,
    cache: bool = False
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a list of TFRecord files.
    
    This function creates an optimized data pipeline with:
    - Parallel reading of TFRecord files
    - Parallel parsing of examples
    - Optional shuffling for training
    - Batching
    - Prefetching for performance
    
    Parameters
    ----------
    tfrecord_paths : List[str]
        List of paths to TFRecord files.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle the dataset (typically True for training, False for val/test).
    shuffle_buffer_size : int
        Size of the shuffle buffer. Larger values provide better randomness but use more memory.
    cache : bool
        Whether to cache the dataset in memory (useful for small datasets).
    
    Returns
    -------
    tf.data.Dataset
        Batched and prefetched dataset ready for training/evaluation.
    """
    if not tfrecord_paths:
        raise ValueError("No TFRecord paths provided")
    
    # Create dataset from TFRecord files with parallel reading
    dataset = tf.data.TFRecordDataset(
        tfrecord_paths,
        num_parallel_reads=tf.data.AUTOTUNE
    )
    
    # Optional caching (useful if dataset fits in memory)
    if cache:
        dataset = dataset.cache()
    
    # Shuffle before parsing for better randomness
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            reshuffle_each_iteration=True
        )
    
    # Parse TFRecords with parallel processing
    dataset = dataset.map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def verify_dataset_split_csv(tfrecord_dir: str) -> str:
    """
    Verify that dataset_split.csv exists in the tfrecord_dir and return its path.
    
    Parameters
    ----------
    tfrecord_dir : str
        Base directory that should contain dataset_split.csv.
    
    Returns
    -------
    str
        Path to dataset_split.csv.
    
    Raises
    ------
    FileNotFoundError
        If dataset_split.csv is not found.
    """
    csv_path = Path(tfrecord_dir) / "dataset_split.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"dataset_split.csv not found in {tfrecord_dir}. "
            "Make sure you've run the preprocessing pipeline first."
        )
    
    return str(csv_path)


def get_dataset_info(tfrecord_dir: str) -> dict:
    """
    Extract dataset information from dataset_split.csv for logging purposes.
    
    Parameters
    ----------
    tfrecord_dir : str
        Base directory containing dataset_split.csv.
    
    Returns
    -------
    dict
        Dictionary containing dataset statistics:
        - total_samples: Total number of samples
        - train_samples: Number of training samples
        - val_samples: Number of validation samples
        - test_samples: Number of test samples
        - num_classes: Number of unique classes
        - class_distribution: Distribution of classes across splits
    """
    csv_path = verify_dataset_split_csv(tfrecord_dir)
    df = pd.read_csv(csv_path)
    
    info = {
        'total_samples': len(df),
        'train_samples': len(df[df['split'] == SplitLabels.TRAIN]),
        'val_samples': len(df[df['split'] == SplitLabels.VAL]),
        'test_samples': len(df[df['split'] == SplitLabels.TEST]),
        'num_classes': df['label'].nunique(),
        'csv_path': csv_path
    }
    
    # Class distribution per split
    class_dist = {}
    for split_name, split_value in [('train', SplitLabels.TRAIN), 
                                     ('val', SplitLabels.VAL), 
                                     ('test', SplitLabels.TEST)]:
        split_df = df[df['split'] == split_value]
        class_dist[split_name] = split_df['label'].value_counts().to_dict()
    
    info['class_distribution'] = class_dist
    
    return info


# ============================================================================
# MODEL COMPILATION
# ============================================================================

def compile_model_if_needed(
    model: tf.keras.Model,
    learning_rate: float = TrainingConstants.LEARNING_RATE
) -> tf.keras.Model:
    """
    Compile the model if it hasn't been compiled yet.
    
    This function checks if the model is already compiled and only compiles
    if necessary. Uses sparse categorical crossentropy since labels are integers.
    
    Parameters
    ----------
    model : tf.keras.Model
        The model to compile.
    learning_rate : float
        Learning rate for the Adam optimizer.
    
    Returns
    -------
    tf.keras.Model
        The compiled model.
    """
    # Check if model is already compiled by checking if optimizer exists
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        log.info(f"Compiling model with learning_rate={learning_rate}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        log.info("Model is already compiled, skipping compilation")
    
    return model


# ============================================================================
# MODEL SAVING AND LOGGING
# ============================================================================

def save_model_and_log(
    model: tf.keras.Model,
    model_save_dir: str,
    model_name: str,
    csv_log_path: str = ModelsCSV.TRAINING,
    dataset_csv_path: str = os.path.join(DATA_SPLIT_CSV_PATH, "dataset_split.csv"),
    epochs_trained: int = TrainingConstants.EPOCHS,
    batch_size: int = TrainingConstants.BATCH_SIZE,
    learning_rate: float = TrainingConstants.LEARNING_RATE,
    val_loss: Optional[float] = None,
    val_accuracy: Optional[float] = None,
    train_loss: Optional[float] = None,
    train_accuracy: Optional[float] = None,
    notes: str = "",
) -> str:
    """
    Save a trained TensorFlow model and log metadata to a CSV registry.
    
    This function:
    1. Saves the model with a timestamp in the filename
    2. Logs training metadata to a CSV file for experiment tracking
    3. Returns the path where the model was saved
    
    The CSV registry allows tracking of all trained models with their
    hyperparameters and performance metrics for comparison and reproducibility.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained TensorFlow model to save.
    model_save_dir : str
        Directory where the model will be saved.
    model_name : str
        Human-readable model name (used in filename and logging).
    csv_log_path : str
        Path to the CSV file storing model metadata registry.
    dataset_csv_path : str
        Path to the dataset_split.csv used for training.
    epochs_trained : int
        Number of epochs actually trained (may be less than max if early stopped).
    batch_size : int
        Training batch size.
    learning_rate : float
        Learning rate used for training.
    val_loss : float, optional
        Best validation loss achieved during training.
    val_accuracy : float, optional
        Best validation accuracy achieved during training.
    train_loss : float, optional
        Final training loss.
    train_accuracy : float, optional
        Final training accuracy.
    notes : str
        Free text notes about the experiment (architecture changes, hyperparameters, etc.).
    
    Returns
    -------
    str
        Path to the saved model.
    """
    # Generate timestamp for unique model identification
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create model save directory if it doesn't exist
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Save the model with timestamp
    model_path = model_save_dir / f"{model_name}.h5"
    model.save(model_path)
    log.info(f"Model saved to: {model_path}")

    # Prepare log entry with all relevant information
    log_entry = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_path": str(model_path),
        "dataset_csv_path": dataset_csv_path,
        "epochs_trained": epochs_trained,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "framework": "tensorflow",
        "notes": notes,
    }

    # Append to CSV registry
    csv_log_path = Path(csv_log_path)
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_log_path.exists():
        df = pd.read_csv(csv_log_path)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(csv_log_path, index=False)
    log.info(f"Training metadata logged to: {csv_log_path}")
    
    return str(model_path)


# ============================================================================
# UNIFIED TRAINING PIPELINE
# ============================================================================

def train_model_pipeline_04(
    model: tf.keras.Model,
    tfrecord_dir: str,
    batch_size: int = TrainingConstants.BATCH_SIZE,
    epochs: int = TrainingConstants.EPOCHS,
    learning_rate: float = TrainingConstants.LEARNING_RATE,
    early_stopping_patience: int = TrainingConstants.EARLY_STOPPING_PATIENCE,
    save: bool = False,
    model_save_dir: Optional[str] = ModelsCSV.TRAINING_DIR,
    model_registry_csv: Optional[str] = ModelsCSV.TRAINING,
    notes: str = "",
    cache_dataset: bool = False,
    shuffle_buffer_size: int = 2000,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Complete end-to-end pipeline for training a music genre classification model.
    
    This unified pipeline handles:
    1. Verifying the data structure and dataset_split.csv
    2. Loading TFRecords from train/val/test subdirectories
    3. Building optimized tf.data.Dataset pipelines
    4. Compiling the model if needed
    5. Training with early stopping
    6. Optionally saving the model and logging to a registry
    
    The new data structure expects:
        tfrecord_dir/
            train/*.tfrecord       - Training TFRecords
            val/*.tfrecord         - Validation TFRecords
            test/*.tfrecord        - Test TFRecords
            dataset_split.csv      - Metadata (path, label, split)
            normalization_stats.json - Normalization parameters
    
    Parameters
    ----------
    model : tf.keras.Model
        The model to train. Can be compiled or uncompiled.
    tfrecord_dir : str
        Base directory containing train/val/test subdirectories and dataset_split.csv.
    batch_size : int
        Number of samples per batch.
    epochs : int
        Maximum number of training epochs.
    learning_rate : float
        Learning rate for the Adam optimizer.
    early_stopping_patience : int
        Number of epochs with no improvement after which training will be stopped.
    save : bool
        If True, save the trained model and log metadata to the registry.
    model_save_dir : str, optional
        Directory to save the trained model (required if save=True).
    model_registry_csv : str, optional
        Path to CSV file for logging model metadata (required if save=True).
    notes : str
        Free text notes about the experiment (architecture, hyperparameters, etc.).
    cache_dataset : bool
        If True, cache the dataset in memory (useful for small datasets).
    shuffle_buffer_size : int
        Size of the shuffle buffer for training data.
    
    Returns
    -------
    Tuple[tf.keras.Model, tf.keras.callbacks.History]
        - Trained model with best weights restored
        - Training history object containing loss and metrics
    
    Raises
    ------
    FileNotFoundError
        If dataset_split.csv is not found in tfrecord_dir.
    ValueError
        If save=True but required paths are not provided.
    
    Example
    -------
    >>> from tensorflow.keras import models, layers
    >>> 
    >>> # Define a simple CNN model
    >>> model = models.Sequential([
    ...     layers.Input(shape=(128, 130, 1)),
    ...     layers.Conv2D(32, 3, activation='relu'),
    ...     layers.MaxPooling2D(2),
    ...     layers.Flatten(),
    ...     layers.Dense(10, activation='softmax')
    ... ])
    >>> model.name = "simple_cnn"
    >>> 
    >>> # Train with the pipeline
    >>> trained_model, history = train_model_pipeline(
    ...     model=model,
    ...     tfrecord_dir="data/tfrecords",
    ...     batch_size=32,
    ...     epochs=50,
    ...     save=True,
    ...     notes="Baseline CNN model"
    ... )
    """
    
    log.info("="*80)
    log.info("STARTING MODEL TRAINING PIPELINE")
    log.info("="*80)
    
    # ========================================================================
    # Step 1: Verify data structure and get dataset info
    # ========================================================================
    log.info("Step 1/6: Verifying data structure...")
    
    try:
        dataset_info = get_dataset_info(tfrecord_dir)
        log.info(f"Dataset info:")
        log.info(f"  - Total samples: {dataset_info['total_samples']}")
        log.info(f"  - Train samples: {dataset_info['train_samples']}")
        log.info(f"  - Val samples: {dataset_info['val_samples']}")
        log.info(f"  - Test samples: {dataset_info['test_samples']}")
        log.info(f"  - Number of classes: {dataset_info['num_classes']}")
    except FileNotFoundError as e:
        log.error(str(e))
        raise
    
    # ========================================================================
    # Step 2: Load TFRecord paths from subdirectories
    # ========================================================================
    log.info("Step 2/6: Loading TFRecord paths...")
    
    train_paths = get_tfrecord_paths_from_split(tfrecord_dir, "train")
    val_paths = get_tfrecord_paths_from_split(tfrecord_dir, "val")
    
    if not train_paths:
        raise ValueError(f"No training TFRecords found in {tfrecord_dir}/train/")
    if not val_paths:
        raise ValueError(f"No validation TFRecords found in {tfrecord_dir}/val/")
    
    # ========================================================================
    # Step 3: Build tf.data.Dataset pipelines
    # ========================================================================
    log.info("Step 3/6: Building data pipelines...")
    
    train_ds = build_dataset_from_tfrecords(
        tfrecord_paths=train_paths,
        batch_size=batch_size,
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        cache=cache_dataset
    )
    
    val_ds = build_dataset_from_tfrecords(
        tfrecord_paths=val_paths,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        cache=cache_dataset
    )
    
    log.info(f"Training batches per epoch: ~{dataset_info['train_samples'] // batch_size}")
    log.info(f"Validation batches per epoch: ~{dataset_info['val_samples'] // batch_size}")
    
    # ========================================================================
    # Step 4: Compile model if needed
    # ========================================================================
    log.info("Step 4/6: Preparing model...")
    
    model = compile_model_if_needed(model, learning_rate=learning_rate)
    
    # Log model summary
    log.info(f"Model name: {model.name if hasattr(model, 'name') else 'unnamed'}")
    log.info(f"Total parameters: {model.count_params():,}")
    
    # ========================================================================
    # Step 5: Train the model
    # ========================================================================
    log.info("Step 5/6: Training model...")
    log.info(f"Training configuration:")
    log.info(f"  - Batch size: {batch_size}")
    log.info(f"  - Max epochs: {epochs}")
    log.info(f"  - Learning rate: {learning_rate}")
    log.info(f"  - Early stopping patience: {early_stopping_patience}")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Extract training results
    epochs_trained = len(history.history["loss"])
    best_val_loss = min(history.history["val_loss"])
    best_val_accuracy = max(history.history.get("val_accuracy", [0]))
    final_train_loss = history.history["loss"][-1]
    final_train_accuracy = history.history.get("accuracy", [0])[-1]
    
    log.info(f"Training completed!")
    log.info(f"  - Epochs trained: {epochs_trained}")
    log.info(f"  - Best validation loss: {best_val_loss:.4f}")
    log.info(f"  - Best validation accuracy: {best_val_accuracy:.4f}")
    log.info(f"  - Final training loss: {final_train_loss:.4f}")
    log.info(f"  - Final training accuracy: {final_train_accuracy:.4f}")
    
    # ========================================================================
    # Step 6: Save model and log metadata (optional)
    # ========================================================================
    if save:
        log.info("Step 6/6: Saving model and logging metadata...")
        
        # Validate required parameters
        if not all([model_save_dir, model_registry_csv]):
            raise ValueError(
                "model_save_dir and model_registry_csv must be provided when save=True"
            )
        
        if not hasattr(model, 'name') or not model.name:
            log.warning("Model has no name, using 'unnamed_model'")
            model.name = 'unnamed_model'
        
        # Save model and log to registry
        model_path = save_model_and_log(
            model=model,
            model_save_dir=model_save_dir,
            model_name=model.name,
            csv_log_path=model_registry_csv,
            dataset_csv_path=dataset_info['csv_path'],
            epochs_trained=epochs_trained,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_loss=best_val_loss,
            val_accuracy=best_val_accuracy,
            train_loss=final_train_loss,
            train_accuracy=final_train_accuracy,
            notes=notes,
        )
        
        log.info(f"Model saved to: {model_path}")
    else:
        log.info("Step 6/6: Skipping model save (save=False)")
    
    log.info("="*80)
    log.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    log.info("="*80)
    
    return model, history


# ============================================================================
# EVALUATION FUNCTION (BONUS)
# ============================================================================

def evaluate_model_on_test(
    model: tf.keras.Model,
    tfrecord_dir: str,
    batch_size: int = TrainingConstants.BATCH_SIZE
) -> dict:
    """
    Evaluate a trained model on the test set.
    
    This function loads the test TFRecords and evaluates the model's
    performance on held-out data that was never seen during training.
    
    Parameters
    ----------
    model : tf.keras.Model
        Trained model to evaluate.
    tfrecord_dir : str
        Base directory containing the test subdirectory.
    batch_size : int
        Batch size for evaluation.
    
    Returns
    -------
    dict
        Dictionary containing test metrics:
        - test_loss: Loss on test set
        - test_accuracy: Accuracy on test set
        - num_test_samples: Number of test samples evaluated
    """
    log.info("="*80)
    log.info("EVALUATING MODEL ON TEST SET")
    log.info("="*80)
    
    # Get test TFRecord paths
    test_paths = get_tfrecord_paths_from_split(tfrecord_dir, "test")
    
    if not test_paths:
        raise ValueError(f"No test TFRecords found in {tfrecord_dir}/test/")
    
    # Build test dataset
    test_ds = build_dataset_from_tfrecords(
        tfrecord_paths=test_paths,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle test data
        cache=False
    )
    
    # Evaluate
    log.info(f"Evaluating on {len(test_paths)} test TFRecord files...")
    results = model.evaluate(test_ds, verbose=1)
    
    # Extract metrics (model.evaluate returns [loss, accuracy, ...])
    test_loss = results[0]
    test_accuracy = results[1] if len(results) > 1 else None
    
    log.info(f"Test Results:")
    log.info(f"  - Test loss: {test_loss:.4f}")
    if test_accuracy is not None:
        log.info(f"  - Test accuracy: {test_accuracy:.4f}")
    
    log.info("="*80)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'num_test_samples': len(test_paths)
    }