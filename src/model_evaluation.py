from typing import Dict
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from src.logger import get_logger

log = get_logger("model_evaluation")


def evaluate_and_log_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    training_metadata: Dict,
    model_metadata: Dict,
    eval_csv_path: str
) -> None:
    """
    Evaluate a trained model, compute metrics and append results to CSV.

    Args:
        model : tf.keras.Model
            Trained model.
        dataset : tf.data.Dataset
            Validation or test dataset.
        training_metadata : Dict
            Metadata returned by train_model.
        model_metadata : Dict
            Model description:
                - model_id: str
                - generation: int
                - num_params: int
        eval_csv_path : str
            Path to evaluation CSV file.

    Returns:
        None
    """
    y_true = []
    y_pred = []

    for x, y in dataset:
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    gap = training_metadata["loss_val"] - training_metadata["loss_train"]

    # Example fitness function (can be changed later)
    fitness = f1_macro - 0.3 * abs(gap)

    row = {
        "model_id": model_metadata["model_id"],
        "generation": model_metadata["generation"],
        "fitness": fitness,
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        "loss_train": training_metadata["loss_train"],
        "loss_val": training_metadata["loss_val"],
        "gap": gap,
        "epochs_trained": training_metadata["epochs_trained"],
        "num_params": model_metadata["num_params"],
        "train_time_sec": training_metadata["training_time_sec"],
        "status": "OK"
    }

    eval_csv = Path(eval_csv_path)
    eval_csv.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([row])
    if eval_csv.exists():
        df_row.to_csv(eval_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(eval_csv, index=False)

    log.info(
        f"Model {model_metadata['model_id']} evaluated: "
        f"fitness={fitness:.4f}, f1_macro={f1_macro:.4f}"
    )
