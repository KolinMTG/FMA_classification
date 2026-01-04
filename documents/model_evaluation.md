
## 2.2 Global Metrics

| Name            | Var         | Role                 |
| --------------- | ----------- | -------------------- |
| Accuracy        | acc         | Raw performance      |
| Macro F1        | f1_macro    | Inter-class fairness |
| Weighted F1     | f1_weighted | Robustness           |
| Validation loss | loss_val    | Stability            |
| Overfitting gap | gap         | Generalization       |

---

## 2.3 Dynamic metrics (critical for GA)

These metrics allow evaluating a model that is only partially trained.

| Name              | Description                                 |
| ----------------- | ------------------------------------------- |
| epochs_trained    | Number of effective epochs                  |
| best_epoch        | Epoch with best validation result           |
| convergence_speed | Epoch where validation loss reaches minimum |
| divergence_flag   | Loss explosion / NaN                        |
| learning_slope    | Loss derivative over the first epochs       |

---

## Fitness Value

Definition of α, β, γ, σ:

**fitness = α·f1_macro − β·gap − γ·loss_val − σ·complexity**

* Favors stability
* Penalizes complexity and overfitting

---

## Storing results in a CSV with precise fields

```
model_id,generation,fitness,f1_macro,accuracy,loss_val,loss_train,gap,epochs_trained,num_params,train_time_sec,status
```

| Field name       | Type    | Description                                                                                                        |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `model_id`       | `str`   | Unique identifier of the evaluated model. Links results to a specific architecture, configuration, or genome.      |
| `generation`     | `int`   | Generation index in the optimization algorithm (e.g., genetic algorithm). Useful for analyzing progress over time. |
| `fitness`        | `float` | Global score used for model selection. Weighted combination of metrics (F1, overfitting, complexity, etc.).        |
| `f1_macro`       | `float` | Macro-averaged F1 score on the validation set. Measures balance between classes.                                   |
| `accuracy`       | `float` | Global accuracy on the validation set. Indicates the proportion of correct predictions.                            |
| `loss_val`       | `float` | Minimum validation loss. Indicator of generalization ability.                                                      |
| `loss_train`     | `float` | Minimum training loss. Used to detect overfitting.                                                                 |
| `gap`            | `float` | Difference `loss_val - loss_train`. Direct measure of model overfitting.                                           |
| `epochs_trained` | `int`   | Actual number of epochs completed (including early stopping). Indicator of convergence speed.                      |
| `num_params`     | `int`   | Total number of trainable parameters. Used to penalize overly complex models.                                      |
| `train_time_sec` | `float` | Total training time in seconds. Allows incorporating computation cost.                                             |
| `status`         | `str`   | Final evaluation status (`OK`, `FAILED`, `DIVERGED`). Helps filter invalid models.                                 |


