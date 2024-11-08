import os
import datetime as dt
import pandas as pd
import numpy as np

from src.constants import DATA_DIR_NAME

NUM_DAYS_IN_WEEK = 7


def load_data(file_name: str) -> pd.DataFrame:
    """
    Load a CSV data file from the 'data' directory.

    """
    if not isinstance(file_name, str):
        raise TypeError("The input 'file_name' must be a string.")

    package_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(package_dir, DATA_DIR_NAME)
    file_path = os.path.join(data_dir, file_name)

    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file '{file_name}' not found in directory '{data_dir}'.") from e


def map_into_numeric(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Map categorical values in a DataFrame to numeric values based on provided mappings.

    Example:
        If mappings = {'color': {'red': 1, 'blue': 2}} and 'color' is a column in the DataFrame,
        the function will replace 'red' with 1 and 'blue' with 2 in the 'color' column.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")

    if not isinstance(mappings, dict):
        raise TypeError("The input 'mappings' must be a dictionary.")

    for col_name in mappings:
        df = df.replace({col_name: mappings[col_name]})
    return df


def is_data_full(df: pd.DataFrame) -> bool:
    """
    Check if the given DataFrame spans consecutive days without any missing dates.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    try:
        first_day = dt.datetime.strptime(df.index[0], '%Y-%m-%d')
        last_day = dt.datetime.strptime(df.index[-1], '%Y-%m-%d')
    except ValueError:
        raise ValueError("The index of the DataFrame must contain dates in the format '%Y-%m-%d'.")

    return first_day + dt.timedelta(len(df) - 1) == last_day


def preprocess_data(file_name: str, y_col, prc_samples_for_test,
                    numeric_mappings):
    try:
        df = load_data(file_name)  # Assumes CSV format, change if necessary
    except FileNotFoundError:
        raise FileNotFoundError(
            "File not found. Please provide a valid file name.")
    state_holiday_mapping = {'0': 0, 'a': 1, 'b': 1, 'c': 1}
    df = map_into_numeric(df, mappings=numeric_mappings)
    df = df.groupby('Date').mean()
    df.drop(['Store'], axis=1, inplace=True)
    assert is_data_full(df), "Data is not full."

    n_samples_test = int(prc_samples_for_test * len(df))
    train_df = df[:-n_samples_test]
    test_df = df[-n_samples_test:]

    y_train = train_df[y_col].squeeze()
    y_test = test_df[y_col].squeeze()

    X_train = train_df.drop(columns=[y_col])
    X_test = test_df.drop(columns=[y_col])

    return X_train, y_train, X_test, y_test
