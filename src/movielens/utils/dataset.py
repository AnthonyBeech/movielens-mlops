import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


def remove_nulls(df: pd.DataFrame, subset: list | None = None) -> pd.DataFrame:
    """Remove rows with null values from the DataFrame."""
    df = df.dropna(subset=subset).reset_index(drop=True)
    log.debug(f"After dropping nulls: {len(df)}")
    return df


def keep_by_value(
    df: pd.DataFrame, col: str, min_value: float | None = None, max_value: float | None = None
) -> pd.DataFrame:
    """Keep values in column by min/max."""
    df = df[df[col] > min_value] if min_value is not None else df
    df = df[df[col] < max_value] if max_value is not None else df
    log.debug(f"After removing by range: {len(df)}")


def keep_by_count(
    df: pd.DataFrame, col: str, min_count: float | None = None, max_count: float | None = None
) -> pd.DataFrame:
    """Keep rows from the DataFrame based on the frequency count of values in a given column."""
    counts = df[col].value_counts()

    valid = counts[counts > min_count] if min_count is not None else counts
    valid = valid[valid < max_count] if max_count is not None else valid
    valid_values = valid.index

    cleaned_df = df[df[col].isin(valid_values)].reset_index(drop=True)
    log.debug(f"After remove by count: {len(df)}")
    return cleaned_df


def balance_col(df: pd.DataFrame, col: str, random_state: int) -> pd.DataFrame:
    """Balance the dataset by rating_col."""
    grouped = df.groupby(col)

    target_count = grouped.size().min()

    balanced_groups = [group.sample(n=target_count, random_state=random_state) for _, group in grouped]

    balanced_df = pd.concat(balanced_groups).reset_index(drop=True)
    return balanced_df


def load_data(path: str, n: int | None = None) -> pd.DataFrame:
    """Load movielens data to df. Ratings by default."""
    log.info("loading data")
    df = pd.read_csv(path)[:]

    if n:
        return df[:n]
    return df


def write_data(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Write movielens data to df. Ratings by default."""
    log.info("writing data")
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            msg = "Could not delete original csv"
            log.exception(msg)
            raise
    df.to_csv(path, index=False)


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
    log.info(f"Extraction complete. Files have been unzipped to: {extract_dir}")


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
        log.warning(f"File already exists at {file_path}. Skipping download.")
        return

    log.info(f"Downloading {filename} from {data_link}...")
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

    log.info(f"Download complete. File saved to {file_path}")

    unzip_file(file_path, file_path.parent)
