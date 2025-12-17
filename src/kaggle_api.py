
import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from src.logger import get_logger

log = get_logger(__name__)



def authenticate_kaggle() -> KaggleApi:
    load_dotenv()

    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")

    if not username or not api_key:
        log.error("Kaggle credentials not found in .env file")
        raise RuntimeError("Missing Kaggle credentials in .env file")

    api = KaggleApi()
    api.authenticate()
    log.info("Authenticated to Kaggle API")
    return api


def download_kaggle_resource(
    api: KaggleApi,
    dataset_ref: str,
    output_dir: str,
    file_name: str | None = None,
    unzip: bool = True
) -> None:
    """
    Download a Kaggle dataset or a specific file if it does not already exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading from Kaggle dataset:", dataset_ref)
    print("Output directory:", output_dir)
    print("File name:", file_name if file_name else "Entire dataset")
    api.dataset_download_file(
        dataset=dataset_ref,
        file_name=file_name,
        path=r"C:\Users\colin\Documents\ETUDE\MAIN\UTC semestre 5  PK\Neural Networks\final_project\data\FMA_small\000"
    )

    return True

def list_files(api: KaggleApi) -> None:
    files = api.dataset_list_files("imsparsh/fma-free-music-archive-small-medium").files
    for f in files[:20]:
        print(f)


if __name__ == "__main__":
    api = authenticate_kaggle()
    list_files(api)