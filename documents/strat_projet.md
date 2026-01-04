
# Strategy for the Project

## 1. Fundamental constraints to consider

Before defining any strategy, we need to clearly state the real constraints.

### Hardware constraints

* Limited CPU (i7 11th gen laptop)
* No local GPU
* Limited memory (16 GB)
* Significant computation time per training run

*This implies:*

* partial training
* early elimination strategies
* no exhaustive grid search

### Data constraints

* audio is expensive to process
* dataset is large

*This implies:*

* preprocessing must be done once and frozen
* strict separation between train / val / test

### Scientific constraints

* avoid overfitting
* avoid data leakage
* metrics must be comparable across experiments

---

## 2. What can actually be optimized

It is important to distinguish what is structural from what is tunable.

### Not optimizable (fixed once)

* type of audio representation (log-Mel)
* segment duration
* sampling rate
* split protocol

### Optimizable

* CNN architecture
* learning rate
* batch size
* regularization
* learning rate scheduler
* model depth

---

## 3. Main stages of the project

* audio extraction → once
* TFRecord conversion → once
* train/val/test split → once

Everything else should be iterative.

---

## 4. Planned global pipeline (overview)

### Step 0 — Cleaning the initial dataset

Select the audio to be used first.

* choosing classes
* checking existing files
* strict balancing
* label ↔ index mapping
* final CSV generation

**Implemented by: [data_utils](../src/data_utils.py)** 

---

### Step 1 — Feature extraction for selected audio

* TFRecord generation
* the dataset becomes fixed

**Implemented by: [data_pretreat.py](../src/data_pretreat.py)**

---

### Step 2 — Final dataset split

**Goal: ensure clean evaluation.**

* split into train / val / test
* test set never used during optimization
* store split indices

**Implemented by:  [data_pretreat.py](../src/data_pretreat.py)**

---

### Step 3 — Define the search space

Before training anything, define:

* realistic bounds
* discrete vs continuous parameters
* constraints (e.g., maximum model size)

**See file: [research_range.md](research_range.md)**

---

### Step 4 — Baseline model

Goal: provide a comparison reference.

* simple CNN
* full training
* reference metrics

**Implemented by: [cste.py](../src/cste.py) and [model_generator.py](../src/model_generator.py)**

---

### Step 5 — Fast evaluation (fitness proxy)

For each candidate:

* partial training
* using train + val
* aggressive early stopping
* few epochs

The metric is not final performance, but:
the ability to learn quickly without diverging.

**Implemented by : [model_generator.py](../src/model_generator.py), [model_evaluation.py](../src/model_evaluation.py) and [model_optimization.py](../src/model_optimization.py)**

---

### Step 6 — Hyperparameter optimization

Regardless of the algorithm (GA, Hyperband, PBT), the logic is the same:

* generate candidates
* evaluate quickly
* eliminate the worst
* focus resources on the most promising ones

**IMPORTANT: log every experiment and version all tested configs**
**Implemented by : [model_generator.py](../src/model_generator.py), [model_evaluation.py](../src/model_evaluation.py), [optimisation_strats.py](../src/optimisation_strats.py) and [model_optimization.py](../src/model_optimization.py)** 

---

### Step 7 — Final selection

Select a stable architecture for the project.
The choice must be justified (not only best accuracy).

Reminder: Stability > raw performance.
A slightly weaker model but stable is preferable.

**Implemented by : [model_generator.py](../src/model_generator.py), [model_evaluation.py](../src/model_evaluation.py) and [model_optimization.py](../src/model_optimization.py)**

---

### Step 8 — Final long training

Only now:

* full dataset
* higher number of epochs
* full callbacks
* checkpoints
* detailed monitoring

This is the only “expensive” training.

**Implemented by : [model_generator.py](../src/model_generator.py), [model_evaluation.py](../src/model_evaluation.py) and [model_optimization.py](../src/model_optimization.py)**

---

### Step 9 — Final evaluation on the test set

Absolute rule:

The test set is used only once.

You produce:

* accuracy
* confusion matrix
* precision / recall per class


**Implemented by : [model_generator.py](../src/model_generator.py), [model_evaluation.py](../src/model_evaluation.py) and [model_optimization.py](../src/model_optimization.py)**



### Step 10  : Inference on new data

From a new .mp3 file, be able to pretreate it, extract data from audio file according to a prevous pretreatement pipeline to fit training data. Do a prediction on a trained model and produce a report for a evaluated data. 

**Implemented by  : [inference.py](../src/inference.py)**

### Step 11 : Main pipeline 

Put all the prevous step in an automatic pipeline. 

**Implemented by : [main_pipeline.py](../src/main_pipeline.py)**
