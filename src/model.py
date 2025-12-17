# model.py
from typing import List, Optional, Dict, Any
import tensorflow as tf
from src.logger import get_logger
from src.cste import *

log = get_logger("model.log")

# -----------------------------
# Baseline CNN
# -----------------------------
def build_baseline_model(input_shape: tuple = (128, 128, 1), num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build a simple CNN baseline for log-Mel spectrogram classification.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input spectrogram (H, W, C)
    num_classes : int
        Number of target classes

    Returns
    -------
    tf.keras.Model
        Compiled CNN model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    log.info(f"Baseline model built with input_shape={input_shape} and num_classes={num_classes}")
    return model


# -----------------------------
# Flexible / Tuned CNN
# -----------------------------
def build_tuned_model(
    input_shape: tuple = (128, 128, 1),
    num_classes: int = 6,
    conv_layers: int = 3,
    filters: Optional[List[int]] = None,
    kernel_sizes: Optional[List[int]] = None,
    pool_sizes: Optional[List[int]] = None,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    optimizer_name: str = 'adam'
) -> tf.keras.Model:
    """
    Build a flexible CNN for hyperparameter tuning.

    Parameters
    ----------
    input_shape : tuple
        Input shape (H, W, C)
    num_classes : int
        Number of output classes
    conv_layers : int
        Number of convolutional blocks
    filters : List[int], optional
        List of number of filters per convolutional layer. Default doubles each layer if None
    kernel_sizes : List[int], optional
        List of kernel sizes per conv layer. Default 3
    pool_sizes : List[int], optional
        List of pool sizes per layer. Default 2
    dense_units : int
        Units in the fully connected layer
    dropout_rate : float
        Dropout rate after dense layer
    learning_rate : float
        Learning rate for optimizer
    optimizer_name : str
        Optimizer to use ('adam', 'rmsprop', 'sgd')

    Returns
    -------
    tf.keras.Model
        Compiled CNN model
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Set defaults
    if filters is None:
        filters = [32 * (2 ** i) for i in range(conv_layers)]
    if kernel_sizes is None:
        kernel_sizes = [3] * conv_layers
    if pool_sizes is None:
        pool_sizes = [2] * conv_layers

    for i in range(conv_layers):
        x = tf.keras.layers.Conv2D(filters[i], kernel_size=(kernel_sizes[i], kernel_sizes[i]), 
                                   activation='relu', padding='same', name=f'conv_{i+1}')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i]), name=f'pool_{i+1}')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tuned_cnn")

    # Choose optimizer
    optimizers = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate)
    }
    optimizer = optimizers.get(optimizer_name.lower(), tf.keras.optimizers.Adam(learning_rate=learning_rate))

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    log.info(f"Tuned model built: conv_layers={conv_layers}, filters={filters}, dense_units={dense_units}, dropout={dropout_rate}, optimizer={optimizer_name}")
    return model
