import os

import pandas as pd

from src.utils import DATA_DIR_NAME


def load_data(file_name: str) -> pd.DataFrame:
    """Load a CSV data file from the 'data' directory."""
    if not isinstance(file_name, str):
        raise TypeError("The input 'file_name' must be a string.")

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_dir, DATA_DIR_NAME)
    file_path = os.path.join(data_dir, file_name)

    try:
        return pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Data file '{file_name}' not found in directory '{data_dir}'.") from e