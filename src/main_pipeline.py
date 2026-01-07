from src.data_utils import *
from src.cste import *
from data_utils import build_csv_pipeline_00
from data_pretreat import build_data_pipeline_01
from model_generator import build_and_compile_model_03
from model_training import train_model_pipeline_04
from src.optimisation_strats import *
from others import clear_folder
import pandas as pd


def main_pipeline() -> None:
    """"""
    #! 00 Create dataset to pretreat to extract features from audio files
    build_csv_pipeline_00()  # ? in module src/data_utils.py

    # # # #! 01 Pretreat dataset to extract features and build TFRecord files
    # # # #! Split the dataset into train, val, test sets

    dataset_df = pd.read_csv(PATH_LABEL_CSV_PATH)
    build_data_pipeline_01(
        dataset=dataset_df,
        output_dir=TFRECORD_OUTPUT_DIR_64,
        sample_rate=DEFAULT_SAMPLE_RATE,
        num_workers=NUM_WORKERS,
        train_ratio=SplitRatios.TRAIN,
        val_ratio=SplitRatios.VAL,
        per_bin_normalization=True,
        seed=RANDOM_SEED,
        max_stats_samples=None,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        min_db=-60.0,
    )  # ? in module src/data_pretreat_new.py

    #! 05 Optimize model architecture and hyperparameters with optimization pipeline
    # example_2_standard_run()
    example_2_standard_run(tfrecord_dir=TFRECORD_OUTPUT_DIR_64)
    analyze_results()


if __name__ == "__main__":
    main_pipeline()
