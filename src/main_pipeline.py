from src.data_utils import *
from src.cste import *
from data_utils import build_csv_pipeline_00
from data_pretreat import build_data_pipeline_01
from model_generator import build_and_compile_model_03
from model_training import train_model_pipeline_04
from others import clear_folder
import pandas as pd




def main_pipeline() -> None:
    """"""
    #! 00 Create dataset to pretreat to extract features from audio files
    # # build_csv_pipeline_00() #? in module src/data_utils.py 

    # # #! 01 Pretreat dataset to extract features and build TFRecord files
    # # #! Split the dataset into train, val, test sets

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
        n_fft= 1024,
        hop_length= 256,
        n_mels=64,
        min_db=-60.0
    ) #? in module src/data_pretreat_new.py

    #! 02 Define searching parameters spaces (check strat_projet.md in documentation folder)

    #! 03 Declare base model (baseline) and train it
    # clear_folder(r"models")

    # model = build_and_compile_model_03(
    #     model_name="cnn_64x86_enhanced_2",
        
    #     input_shape=(32, 43, 1),  # taille des spectrogrammes
    #     output_units=6,           # 6 classes de genres

    #     # Convolution plus profonde et plus de filtres
    #     conv_layers=[
    #         (16, (3, 3)),
    #         (32, (3, 3)),
    #         (64, (3, 3)),
    #     ],
    #     conv_activations=[
    #         "relu",
    #         "relu",
    #         "relu",
    #     ],
    #     pool_size=(2, 2),

    #     # Couches fully connected plus larges
    #     dense_layers=[128, 64],
    #     dense_activations=["relu", "relu"],

    #     dropout_rates=[0.3, 0.3],  # dropout pour régularisation

    #     optimizer="adam",
    #     learning_rate=5e-4,

    #     loss="sparse_categorical_crossentropy",
    #     metrics=["accuracy"],

    #     save=True
    # )

    #! 04 Train the model with training pipeline
    # print("----------------------------- MODEL SUMMARY -----------------------------")
    # print(model.summary())
    # print("-------------------------------------------------------------------------")

    # train_model_pipeline_04(
    #     model=model,
    #     tfrecord_dir=TFRECORD_OUTPUT_DIR_32,
    #     batch_size=TrainingConstants.BATCH_SIZE,
    #     epochs=TrainingConstants.EPOCHS,
    #     learning_rate=TrainingConstants.LEARNING_RATE,
    #     early_stopping_patience=TrainingConstants.EARLY_STOPPING_PATIENCE,
    #     save=True,
    #     model_save_dir=ModelsCSV.TRAINING,
    #     model_registry_csv=ModelsCSV.REGISTRY,
    #     notes="First baseline model training with new pipeline.",
    # )


if __name__ == "__main__":
    main_pipeline()
