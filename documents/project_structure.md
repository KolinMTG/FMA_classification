# Project Structure

This file contains all informations about the project structure. First of all, here the detail of what each folders contains. 


## Data Folder

* **FMA_medium** or **FMA_small** : mutiples `.mp3` files
* **metadata**: contains all data in `.csv` format
* **tfrecords_{*}**: format of a pretreated data folder



### Metadata

#### filtered_tracks.csv

| track_id | genre        |
| -------- | ------------ |
| 19192    | Experimental |

#### Path Label

| path                          | label |
| ----------------------------- | ----- |
| data/FMA_small/124/124755.mp3 | 0     |

### TFRecord Folder (Pretreated Data)

The folders starting with tfrecords_* contain all the data required to train a model after preprocessing the FMA_medium dataset.

When the pipeline runs correctly, three subfolders are generated, each containing .tfrecord files. They are named:
- train — used for training
- val — used for validation
- test — used for testing

Each TFRecord file is stored using a six-digit filename, for example: 000132.tfrecord

Two additional files are also present:

- **dataset_split.csv** Contains the mapping information linking audio samples to their TFRecord files (path, label, split), where:
    - 0 = train
    - 1 = val
    - 2 = test

- **normalization_strats.json** : Stores all preprocessing parameters (normalization, n_mels, hop_length, sample_rate, etc.). This file is required to run inference on any .mp3 after the model has been trained.

### TFRecord (native TensorFlow format)

| Field       | TFRecord Type / Format | Content                                                                                                                        |
| ----------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| spectrogram | bytes_list             | Log-Mel spectrogram of the segment, stored as `float32` and serialized to bytes. Original size (H, W, 1) before serialization. |
| label       | int64_list             | Integer label corresponding to the audio genre.                                                                                |
| height      | int64_list             | Spectrogram height (H) before serialization.                                                                                   |
| width       | int64_list             | Spectrogram width (W) before serialization.                                                                                    |

---

## Source Folder

* **[cste.py](../src/cste.py)**: contains all project constants; ideally these could be split into separate classes for improved clarity.

* **[logger.py](../src/logger.py)**: handles creation, management, and deletion of logs.

* **[others.py](../src/others.py)**: contains helper utilities that are not part of the main pipeline but are useful for experimentation and development. Includes a trash-like historical file deletion system.

* **[data_utils.py](../src/data_utils.py)**: Step 00 — builds the CSV file listing all selected and balanced audio data.

* **[data_pretreat.py](../src/data_pretreat.py)**: Step 01 — extracts and stores fixed TFRecord files used for training.

* **[model_generator.py](../src/model_generator.py)**: contains functionalities related to model creation. Models are saved in the `models` folder inside `data`.

* **[model_training.py](../src/model_training.py)**: contains all functionalities related to training models created with `model_generator` using data prepared through `data_utils` and the preprocessing pipeline.

* **[model_evaluation.py](../src/model_evaluation.py)**: contains functions to evaluate a given TensorFlow model.

* **[model_optimization.py](../src/model_optimization.py)**: uses `model_evaluation` and `model_generator` to optimize a baseline model in two stages:

  1. Random Search
  2. Neuro-evolution

  Returns and evaluates an optimized model for the musical genre classification task.

* **[optimisation_strats.py](../src/optimisation_strats.py)**: contains different optimization pipelines (short or long) depending on available time. Model 1 (the shortest pipeline) still runs for several hours on CPU. Reminder: AI training requires significant compute resources and is difficult to optimize fully.

* **[inference.py](../src/inference.py)**: performs inference on `.mp3` audio files for a given model.

* **[main_pipeline.py](../src/main_pipeline.py)**: orchestrates the full project pipeline — selecting data, building models, training them, evaluating them, selecting the best one, and returning the final model. Also provides a function to run predictions on a given audio file.


## Documents Folder

The document folder contains all Markdown and text-based documentation files for the project.

## .logs and .trash Folders


The **.trash** folders are used to:
- store traces of discarded or experimental code,
- keep records of deleted files that were not moved to the system recycle bin (for traceability),
- optionally store temporary files removed from the data directory or the current working directory.

The **.logs** folder contains all logs generated by the code during execution.

In most cases, logs are written into separate files named after the script or function that produced them.
This makes it easier to track execution history and identify errors more efficiently.