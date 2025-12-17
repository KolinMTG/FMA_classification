from src.data_utils import *
from src.cste import *
from data_pretreat import build_tfrecord_from_dataframe_01




def main_pipeline() -> None:
    """"""
    #! 00 Create dataset to pretreat to extract features from audio files
    build_csv_pipeline_00() #? in module src/data_utils.py 

    #! 01 Pretreat dataset to extract features and build TFRecord files
    dataset_df = pd.read_csv(PATH_LABEL_CSV_PATH)
    build_tfrecord_from_dataframe_01(
        dataset=dataset_df,
        output_dir=TFRECORD_OUTPUT_DIR,
        sample_rate=DEFAULT_SAMPLE_RATE,
        num_workers=NUM_WORKERS
    ) #? in module src/pretreat_pipeline.py

if __name__ == "__main__":
    main_pipeline()
