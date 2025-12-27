import tensorflow as tf
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.cste import *
from src.logger import get_logger

log = get_logger("models")


def _prepare_dropout_rates(hidden_layers, dropout_rate):
    """Ensure dropout_rates is a list with correct length."""
    if isinstance(dropout_rate, float):
        return [dropout_rate] * len(hidden_layers)
    return dropout_rate


def _check_model_name_unique(model_name, model_csv_path):
    """Check that the model name is not already in the CSV registry."""
    csv_path = Path(model_csv_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if model_name in df["model_name"].values:
            log.error(f"Model name '{model_name}' already exists in {model_csv_path}.")
            raise ValueError(f"Model name '{model_name}' already exists in {model_csv_path}.")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)


def _build_model(input_shape, output_units, conv_layers, conv_activations, pool_size, dense_layers, dense_activations, dropout_rates):
    """
    Construct a CNN for spectrograms and return model + layer description.
    Uses GlobalMaxPooling2D to avoid dimension mismatch issues.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    layers_desc = []

    # --- Convolutional blocks ---
    for i, ((filters, kernel_size), activation, dropout) in enumerate(zip(conv_layers, conv_activations, dropout_rates)):
        x = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding="same")(x)
        layers_desc.append(f"conv-{filters}x{kernel_size}-{activation}")
        x = tf.keras.layers.MaxPooling2D(pool_size)(x)
        layers_desc.append(f"maxpool-{pool_size}")
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
            layers_desc.append(f"dropout-{dropout}")

    # --- Global pooling instead of flatten ---
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    layers_desc.append("global_max_pooling2d")

    # --- Dense layers ---
    for neurons, activation in zip(dense_layers, dense_activations):
        x = tf.keras.layers.Dense(neurons, activation=activation)(x)
        layers_desc.append(f"dense-{neurons}-{activation}")

    # --- Output layer ---
    outputs = tf.keras.layers.Dense(output_units, activation="softmax")(x)
    layers_desc.append(f"dense-{output_units}-softmax")

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, layers_desc




def _register_model_csv(model_name, layers_desc, input_shape, output_units, optimizer, learning_rate, model_csv_path):
    """Save model configuration in a CSV registry."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_entry = {
        "timestamp": timestamp,
        "model_name": model_name,
        "layers": ";".join(layers_desc),
        "input_shape": str(input_shape),
        "output_units": output_units,
        "activation_output": "softmax",
        "optimizer": optimizer,
        "learning_rate": learning_rate,
    }

    csv_path = Path(model_csv_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(csv_path, index=False)


def build_and_compile_model_03(
    model_name: str = ModelDefaults.NAME,
    input_shape: tuple = ModelDefaults.INPUT_SHAPE,
    output_units: int = ModelDefaults.OUTPUT_UNITS,
    conv_layers: list = ModelDefaults.CONV_LAYERS,
    conv_activations: list = ModelDefaults.CONV_ACTIVATIONS,
    pool_size: tuple = ModelDefaults.POOL_SIZE,
    dense_layers: list = ModelDefaults.DENSE_LAYERS,
    dense_activations: list = ModelDefaults.DENSE_ACTIVATIONS,
    dropout_rates: list = ModelDefaults.DROPOUT_RATES,
    optimizer: str = ModelDefaults.OPTIMIZER,
    learning_rate: float = ModelDefaults.LEARNING_RATE,
    loss: str = ModelDefaults.LOSS,
    metrics: list[str] = ModelDefaults.METRICS,

    save: bool = True,
    model_csv_path: str = ModelsCSV.REGISTRY,
) -> tf.keras.Model:

    # Step 1: Check name uniqueness
    _check_model_name_unique(model_name, model_csv_path)

    # Step 2: Build CNN model
    model, layers_desc = _build_model(
        input_shape=input_shape,
        output_units=output_units,
        conv_layers=conv_layers,
        conv_activations=conv_activations,
        pool_size=pool_size,
        dense_layers=dense_layers,
        dense_activations=dense_activations,
        dropout_rates=dropout_rates
    )

    # Step 3: Compile
    if not hasattr(model, "optimizer"):
        tf_optimizer = tf.keras.optimizers.get(optimizer)
        if hasattr(tf_optimizer, "learning_rate"):
            tf_optimizer.learning_rate = learning_rate
        model.compile(optimizer=tf_optimizer, loss=loss, metrics=metrics)

    # Step 4: Register CSV
    if save == True:
        _register_model_csv(model_name, layers_desc, input_shape, output_units, optimizer, learning_rate, model_csv_path)

    log.info(f"Model '{model_name}' built, compiled, and registered successfully.")

    return model

