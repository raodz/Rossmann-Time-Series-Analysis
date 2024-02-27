import os
import datetime as dt
import pandas as pd
import numpy as np

NUM_DAYS_IN_WEEK = 7


def load_data(data_file: str):
    """
    Load a CSV data file from the 'data' directory.

    Parameters:
        data_file (str): The name of the CSV data file to load.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified data file is not found.
    """
    package_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(package_dir, 'data')
    file_path = os.path.join(data_dir, data_file)
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file '{data_file}' not found in directory '{data_dir}'.") from e


def map_into_numeric(df: pd.DataFrame, mappings: dict):
    """
    Map categorical values in a DataFrame to numeric values based on provided mappings.

    Parameters:
        df (pd.DataFrame): The DataFrame containing categorical columns to be mapped.
        mappings (dict): A dictionary where keys are column names and values are dictionaries
                         mapping categorical values to numeric values for each column.

    Returns:
        pd.DataFrame: The DataFrame with categorical values mapped to numeric values.

    Example:
        If mappings = {'color': {'red': 1, 'blue': 2}} and 'color' is a column in the DataFrame,
        the function will replace 'red' with 1 and 'blue' with 2 in the 'color' column.

    Note:
        This function does not modify the input DataFrame; it returns a new DataFrame with the mappings applied.

    """
    for col_name in mappings:
        df = df.replace({col_name: mappings[col_name]})
    return df


def is_data_full(data: pd.DataFrame):
    first_day = dt.datetime.strptime(data.index[0], '%Y-%m-%d')
    last_day = dt.datetime.strptime(data.index[-1], '%Y-%m-%d')
    if first_day + dt.timedelta(len(data) - 1) == last_day:
        return True
    else:
        return False


def sliding_window(elements, y_col, window_size: int):
    if len(elements) <= window_size:
        return [], []

    X = {col: [] for col in elements.columns}
    y = []

    for i in range(len(elements) - window_size):
        if elements[y_col][i + window_size] is not None:
            for col in X:
                X[col].append(pd.Series(elements[col][i:i + window_size]))
            y.append(elements[y_col][i + window_size])
    return X, y


def get_X_y(df, input_size=1):
    X, y = sliding_window(df, 'Sales', window_size=NUM_DAYS_IN_WEEK)

    if input_size > 1:
        longer_inputs = {}
        for company in X:
            longer_inputs[company] = []
            for i in range(len(X[company]) - input_size):
                longer_inputs[company].append(np.array(X[company][i:i + input_size]).flatten())
        X = longer_inputs
        y = y[:len(y) - input_size]

    features = []

    for feature in X:
        features.append(
            pd.DataFrame(np.array(X[feature]), columns=[f'{feature}{i}' for i in range(NUM_DAYS_IN_WEEK * input_size)]))

    X = pd.concat(features, axis=1)
    X.index = df.index[NUM_DAYS_IN_WEEK:]
    y = pd.Series(y, index=df.index[NUM_DAYS_IN_WEEK:])
    return X, y
