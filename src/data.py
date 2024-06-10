import os
import datetime as dt
import pandas as pd
import numpy as np
from typing import Tuple

NUM_DAYS_IN_WEEK = 7


def load_data(file_name: str) -> pd.DataFrame:
    """
    Load a CSV data file from the 'data' directory.

    Parameters:
        file_name (str): The name of the CSV data file to load.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified data file is not found.

    Example:
        >>> loaded_data = load_data('example.csv')
        >>> print(loaded_data.head())

    """
    if not isinstance(file_name, str):
        raise TypeError("The input 'file_name' must be a string.")

    package_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(package_dir, 'data')
    file_path = os.path.join(data_dir, file_name)

    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file '{file_name}' not found in directory '{data_dir}'.") from e


def map_into_numeric(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Map categorical values in a DataFrame to numeric values based on provided mappings.

    Parameters:
        df (pd.DataFrame): The DataFrame containing categorical columns to be mapped.
        mappings (dict): A dictionary where keys are column names and values are dictionaries
                         mapping categorical values to numeric values for each column.

    Returns:
        pd.DataFrame: The DataFrame with categorical values mapped to numeric values.

    Raises:
        TypeError: If 'df' is not a pandas DataFrame, or if 'mappings' is not a dictionary.

    Example:
        If mappings = {'color': {'red': 1, 'blue': 2}} and 'color' is a column in the DataFrame,
        the function will replace 'red' with 1 and 'blue' with 2 in the 'color' column.

    Note:
        This function does not modify the input DataFrame; it returns a new DataFrame with the mappings applied.
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

    Parameters:
        df (pd.DataFrame): The DataFrame to be checked. It is assumed that the index of the DataFrame contains dates in the format '%Y-%m-%d'.

    Returns:
        bool: True if the DataFrame spans consecutive days without any missing dates, False otherwise.

    Raises:
        ValueError: If the DataFrame index is not in the expected '%Y-%m-%d' format, or if the DataFrame is empty.

    Example:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> df = pd.DataFrame({'values': [1, 2, 3]}, index=('2024-01-01', '2024-01-02', '2024-01-03'))
        >>> is_data_full(df)
        True
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


def sliding_window(df: pd.DataFrame, y_col: str = 'Sales') -> tuple:
    """
    Generate sliding windows of data for input features (X) and corresponding target values (y).

    This function takes a DataFrame containing both input features and target values and creates sliding windows of the data.
    Sliding windows are created by moving a window of specified size across the DataFrame, extracting input features and
    corresponding target values for each window.

    Parameters:
        df (pd.DataFrame): The DataFrame containing both input features and target values.
        y_col (str): The column name representing the target values in the DataFrame.

    Returns:
        tuple: A tuple containing two lists: X and y.
            X (dict): A dictionary where keys are column names and values are lists of sliding windows for each input feature.
            y (list): A list of target values corresponding to each sliding window.

    Raises:
        ValueError: If the length of the DataFrame is less than or equal to the window size.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10], 'target': [11, 12, 13, 14, 15]})
        >>> X, y = sliding_window(data, 'target', 3)
        >>> print(X)
        {'A': [0    1\n1    2\n2    3\nName: A, dtype: int64, 1    2\n2    3\n3    4\nName: A, dtype: int64], 'B': [0    6\n1    7\n2    8\nName: B, dtype: int64, 1    7\n2    8\n3    9\nName: B, dtype: int64]}
        >>> print(y)
        [14, 15]

    Note:
        The function assumes that the input DataFrame `df` contains consecutive indices.
        The function does not handle missing values or overlapping windows.
    """
    window_size = NUM_DAYS_IN_WEEK

    if len(df) <= window_size:
        print(len(df))
        raise ValueError("The length of the DataFrame must be greater than the window size.")

    X = {col: [] for col in df.columns}
    y = []

    for i in range(len(df) - window_size):
        if df[y_col][i + window_size] is not None:
            for col in X:
                X[col].append(pd.Series(df[col][i:i + window_size]))
            y.append(df[y_col][i + window_size])
    return X, y


def get_X_y(df: pd.DataFrame, y_col: str = 'Sales') -> tuple:
    """
    Generate input features (X) and corresponding target values (y) from the given DataFrame.

    This function prepares the input features and target values for training a predictive model.
    It creates sliding windows of data using the y_col column of the DataFrame and organizes them into X and y.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data, including the y_col column.
        y_col (str): The column name representing the target values in the DataFrame.

    Returns:
        tuple: A tuple containing X and y.
            X (pd.DataFrame or dict): The input features for the model.
            y (pd.Series): The corresponding target values.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'Sales': [100, 120, 130, 110, 105]})
        >>> X, y = get_X_y(data)
        >>> print(X)
           Sales0  Sales1
        2     100     120
        3     120     130
        >>> print(y)
        2    110
        3    105

    """
    X, y = sliding_window(df, y_col)

    features = []

    for feature in X:
        features.append(
            pd.DataFrame(np.array(X[feature]), columns=[f'{feature}{i}' for i in range(NUM_DAYS_IN_WEEK)]))

    X = pd.concat(features, axis=1)
    X.index = df.index[NUM_DAYS_IN_WEEK:]
    y = pd.Series(y, index=df.index[NUM_DAYS_IN_WEEK:])
    return X, y


def preprocess_data(file_name: str, y_col: str = 'Sales', prc_samples_for_test: float = 0.05) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Preprocesses the data by loading, splitting, mapping into numeric values, and preparing for training and testing.

    Parameters:
    - file_name (str): The path to the data file.
    - state_holiday_mapping (dict): A dictionary mapping state holidays to numeric values.
    - prc_samples_for_test (float, optional): The percentage of samples to be used for testing. Default is 0.05.

    Returns:
    - X_train (pd.DataFrame): Features for training.
    - y_train (pd.Series): Target variable for training.
    - X_test (pd.DataFrame): Features for testing.
    - y_test (pd.Series): Target variable for testing.

    Raises:
    - FileNotFoundError: If the specified file cannot be found.
    - AssertionError: If any of the dataframes are not full.
    """
    try:
        df = load_data(file_name)  # Assumes CSV format, change if necessary
    except FileNotFoundError:
        raise FileNotFoundError("File not found. Please provide a valid file name.")

    state_holiday_mapping = {'0': 0, 'a': 1, 'b': 1, 'c': 1}  # All types of holidays are mapped into 1

    n_samples_test = int(prc_samples_for_test * len(df))

    dfs = {'train_df': df[:-n_samples_test], 'test_df': df[-n_samples_test:]}

    for chosen_df in dfs:
        dfs[chosen_df] = map_into_numeric(dfs[chosen_df], {'StateHoliday': state_holiday_mapping})
        dfs[chosen_df] = dfs[chosen_df].groupby('Date').mean()
        dfs[chosen_df].drop(['Store'], axis=1, inplace=True)
        assert is_data_full(dfs[chosen_df]), "Training data is not full."

    X_train, y_train = get_X_y(dfs['train_df'], y_col=y_col)
    X_test, y_test = get_X_y(dfs['test_df'], y_col=y_col)

    return X_train, y_train, X_test, y_test
