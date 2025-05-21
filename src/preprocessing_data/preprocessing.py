from src.preprocessing_data.transformations import map_into_numeric, apply_one_hot_encoding
from src.preprocessing_data.validation import is_data_full
from src.preprocessing_data.filtering import exclude_features, exclude_sundays
from src.preprocessing_data.loading import load_data


def preprocess_data(file_name: str, y_col, prc_samples_for_test,
                    numeric_mappings, excluded_features: list = None,
                    exclude_sundays_flag: bool = False):
    """Preprocess data for time series analysis and split into train/test sets."""
    try:
        df = load_data(file_name)
    except FileNotFoundError:
        raise FileNotFoundError("File not found. Please provide a valid file name.")

    df = map_into_numeric(df, mappings=numeric_mappings)
    df = df.groupby('Date').mean(numeric_only=True)

    if 'Store' in df.columns:
        df.drop(['Store'], axis=1, inplace=True)

    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print("Found NaN values in the following columns:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"{col}: {count} NaN values")

    df = df.fillna(df.mean())
    df = df.reset_index()

    if exclude_sundays_flag:
        print("Excluding Sundays from analysis...")
        df = exclude_sundays(df)

    print("Applying one-hot encoding for days of week...")
    df = apply_one_hot_encoding(df)

    df.set_index('Date', inplace=True)

    if not exclude_sundays_flag:
        assert is_data_full(df), "Data is not complete (has missing dates)."

    if excluded_features:
        df = exclude_features(df, excluded_features)
        print(f"Excluded features: {excluded_features}")
        print(f"Remaining features: {df.columns.tolist()}")

    n_samples_test = int(prc_samples_for_test * len(df))
    train_df = df[:-n_samples_test]
    test_df = df[-n_samples_test:]

    y_train = train_df[y_col].squeeze()
    y_test = test_df[y_col].squeeze()

    X_train = train_df.drop(columns=[y_col])
    X_test = test_df.drop(columns=[y_col])

    return X_train, y_train, X_test, y_test