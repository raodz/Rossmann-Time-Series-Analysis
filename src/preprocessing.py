from src.data import load_data, map_into_numeric, is_data_full, get_X_y


def preprocess_data(file_name: str, state_holiday_mapping: dict, prc_samples_for_test: float = 0.05) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Preprocesses the data by loading, splitting, mapping into numeric values, and preparing for training and testing.

    Args:
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
        df = pd.read_csv(file_name)  # Assumes CSV format, change if necessary
    except FileNotFoundError:
        raise FileNotFoundError("File not found. Please provide a valid file name.")

    n_samples_test = int(prc_samples_for_test * len(df))
    train_df = df[:-n_samples_test]
    test_df = df[-n_samples_test:]

    train_df = map_into_numeric(train_df, {'StateHoliday': state_holiday_mapping})
    train_df = train_df.groupby('Date').mean()
    train_df.drop(['Store'], axis=1, inplace=True)
    assert is_data_full(train_df), "Training data is not full."

    test_df = map_into_numeric(test_df, {'StateHoliday': state_holiday_mapping})
    test_df = test_df.groupby('Date').mean()
    test_df.drop(['Store'], axis=1, inplace=True)
    assert is_data_full(test_df), "Testing data is not full."

    X_train, y_train = get_X_y(train_df, input_size=1)
    X_test, y_test = get_X_y(test_df, input_size=1)

    return X_train, y_train, X_test, y_test
