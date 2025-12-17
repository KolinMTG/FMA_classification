from src.data_utils import *
from src.cste import *
from data_pretreat import build_tfrecord_from_dataframe_01
from data_split import data_split_pipeline_02
from model_generator import build_and_compile_model
from model_training import train_model
from others import clear_folder
import pandas as pd




def main_pipeline() -> None:
    """"""
    #! 00 Create dataset to pretreat to extract features from audio files
    # build_csv_pipeline_00() #? in module src/data_utils.py 

    # #! 01 Pretreat dataset to extract features and build TFRecord files
    # dataset_df = pd.read_csv(PATH_LABEL_CSV_PATH)
    # build_tfrecord_from_dataframe_01(
    #     dataset=dataset_df,
    #     output_dir=TFRECORD_OUTPUT_DIR,
    #     sample_rate=DEFAULT_SAMPLE_RATE,
    #     num_workers=NUM_WORKERS
    # ) #? in module src/pretreat_pipeline.py

    # #! 02 Create splits and build train/val/test CSV file for TFRecord files
    # data_split_pipeline_02(
    #     tfrecord_folder=TFRECORD_OUTPUT_DIR,
    #     output_csv_path=DATA_SPLIT_CSV_PATH,

    #     train_ratio=SplitRatios.TRAIN,
    #     val_ratio=SplitRatios.VAL,
    # ) #? in module src/data_split.py

    #! 03 Define searching parameters spaces (check strat_projet.md in documentation folder)

    #! 04 Declare base model (baseline) and train it
    clear_folder(r"models")
    model = build_and_compile_model()
    print("----------------------------- MODEL SUMMARY -----------------------------")
    print(model.summary())
    print("-------------------------------------------------------------------------")
    train_model(model=model,
                tfrecord_dir=TFRECORD_OUTPUT_DIR,
                csv_path=DATA_SPLIT_CSV_PATH,
                batch_size=TrainingConstants.BATCH_SIZE,
                epochs=TrainingConstants.EPOCHS,
                learning_rate=TrainingConstants.LEARNING_RATE,
                early_stopping_patience=5,
                save=False,
                model_save_dir=ModelsCSV.TRAINING,
                model_registry_csv=ModelsCSV.REGISTRY,
                notes="First baseline model training.",
                )

if __name__ == "__main__":
    main_pipeline()
