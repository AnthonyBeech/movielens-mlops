import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from movielens.config.config import PROJECT_ROOT

from .logger import setup_logging

logger = setup_logging(__name__)


def load_data(path: str = PROJECT_ROOT / "data/movielens/ml-32m/ratings.csv") -> pd.DataFrame:
    """Load movielens data to df. Ratings by default."""
    return pd.read_csv(path)


def unzip_file(zip_path: str, extract_to: str) -> None:
    """
    Extract a ZIP archive into the specified directory.

    Args:
        zip_path (str): The path to the .zip file to be extracted.
        extract_to (str): The directory where the files will be extracted.

    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_to)
    extract_dir.mkdir(parents=True, exist_ok=True)  # Create dirs if they don't exist

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f"Extraction complete. Files have been unzipped to: {extract_dir}")


def get_dataset(
    data_link: str = "https://files.grouplens.org/datasets/movielens/ml-32m.zip", save_dir: str = "data/"
) -> None:
    """
    Download a dataset from the given URL and save it to the specified directory.

    If the file already exists, it won't re-download. Displays a progress bar.

    Args:
        data_link (str): Direct download link to the dataset.
        save_dir (str): Local directory where the file will be saved.

    """
    save_dir_path = Path(save_dir)
    # Create parent directories if they don't exist
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # Infer filename from the URL; fall back to 'dataset.zip' if URL ends with '/'
    filename = data_link.split("/")[-1] or "dataset.zip"
    file_path = save_dir_path / filename

    # If the file already exists, log a warning and skip download
    if file_path.exists():
        logger.warning(f"File already exists at {file_path}. Skipping download.")
        return

    logger.info(f"Downloading {filename} from {data_link}...")
    response = requests.get(data_link, stream=True, timeout=50000)
    response.raise_for_status()  # Raise an HTTPError if status code is 4xx or 5xx

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1KB

    # Handle the case where 'content-length' is missing or 0
    total_chunks = total_size_in_bytes // block_size if total_size_in_bytes else None

    with (
        file_path.open("wb") as file,
        tqdm(desc=filename, total=total_chunks, unit="KB", unit_scale=True) as progress_bar,
    ):
        for data_block in response.iter_content(block_size):
            file.write(data_block)
            progress_bar.update(len(data_block) // block_size)

    logger.info(f"Download complete. File saved to {file_path}")

    unzip_file(file_path, file_path.parent)
