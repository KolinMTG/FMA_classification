# Music Genre Classification from Audio Signals

- **Date**: 2026-01-01
- **Status**: In Progress
- **Author**: Colin MANYRI
- **License**: MIT - Copyright (c) 2026 Colin MANYRI
- **Version**: 10.0.0.1

## Free Music Archive (FMA) – Medium Dataset
This project uses the **FMA Medium (Free Music Archive)** dataset, available on [Kaggle](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium).  

The dataset was introduced by **Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson** at the **ISMIR 2017** conference. It contains over 10,000 audio excerpts of 30 seconds each, with detailed metadata allowing full traceability throughout the experimental pipeline.

- **Project**: Music Genre Classification
- **Type**: Audio signal processing & Machine Learning
- **Primary Language**: Python
- **Project Context**: Academic
- **Dataset Year**: 2025

## Project Objective
This project aims to design and analyze a complete **automatic music genre classification pipeline** from raw audio signals. The focus is on mastering the **entire pipeline**, from loading audio files to evaluating model performance, without using precomputed features.

The project enables building and training AI models using TensorFlow for classification of the main classes of the FMA dataset. Key steps include audio preprocessing, feature extraction, TFRecord generation, baseline model definition, hyperparameter optimization via Random Search and Neuroevolution, and evaluation of the best-performing models.

### Main Features
- Preprocessing and extraction of audio data (MP3 → spectrograms → TFRecord)
- Definition and training of suitable baseline models
- Hyperparameter optimization via Random Search and Neuroevolution
- Evaluation and saving of the most effective models

## Repository Structure
All source code is located in the `src` folder.  
Training data is stored in the `data` folder, which must contain at least **FMA_medium** (the dataset used). The folder may also include:

- `.logs`: project log files  
- `.trash`: files deleted via functions in `src/others.py`  
- Execution-generated folders such as `optimization_results`, `results_quick_test`, `results_tests`, `models`, containing trained models and associated information  

Complete project documentation is located in the `documents` folder, including:

- **model_evaluation.md**: detailed reflections on model evaluation (basis for extending the project)  
- **optimization_documentation.md**: explanation of `model_optimization.py` and `optimisation_strats.py` responsible for baseline model optimization  
- **project_structure.md**: description of files in `src` and their roles, with additional information on project data structures  
- **rearch_range.md**: research range defined for improving the baseline using Random Search and Neuroevolution  
- **strat_projet.md**: planned project steps, providing insight into design thinking prior to coding  
- **tracks_documentation.md**: header format for the `tracks.csv` file located in `data/metadata`  
- **Project_description.pdf**: original academic project brief and grading criteria

## Installation

### Working Environment
- **Recommended IDE**: VS-Code  
- **AI Code Autocompletion**: GitHub Copilot  
- **AI Code Assistance**: Claude Sonnet 4.5, ChatGPT 5.1  
- **Python Version**: 3.10  
- **Environment Manager**: Conda  

#### Create Virtual Environment
```bash
conda create -n FMA python=3.10
conda activate FMA
```
#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Hardware Requirements
Local AI training is resource-intensive (CPU, GPU, and RAM). Recommended minimal configuration:

- CPU: i5-12400f
- RAM: 16 GB DDR4
- GPU: RTX 4060 Ti

## External Elements / Citations

This project uses the FMA Medium (Free Music Archive) dataset, available on Kaggle  [Kaggle](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium)

The dataset was presented by Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson at ISMIR 2017.
**Copyright (c) 2016 Michaël Defferrard**

## Contact / Support / Author

For questions or issues regarding code execution, contact:
Colin MANYRI: colin.manyri@etu.utc.fr