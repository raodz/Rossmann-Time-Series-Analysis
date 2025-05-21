import pandas as pd


def exclude_features(df: pd.DataFrame, excluded_features: list = None) -> pd.DataFrame:
    """Exclude specified features from a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")

    if excluded_features is None or len(excluded_features) == 0:
        return df

    non_existent_features = [f for f in excluded_features if f not in df.columns]
    if non_existent_features:
        print(f"Warning: The following features do not exist in the DataFrame and cannot be excluded: {non_existent_features}")

    features_to_exclude = [f for f in excluded_features if f in df.columns]

    if features_to_exclude:
        return df.drop(columns=features_to_exclude)
    else:
        return df


def exclude_sundays(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude Sundays from the DataFrame based on the DayOfWeek column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input 'df' must be a pandas DataFrame.")

    if 'DayOfWeek' not in df.columns:
        print("Warning: 'DayOfWeek' column not found. Cannot exclude Sundays.")
        return df

    df['DayOfWeek'] = df['DayOfWeek'].astype(int)
    df_no_sundays = df[df['DayOfWeek'] != 7].copy()

    excluded_count = len(df) - len(df_no_sundays)
    print(f"Excluded {excluded_count} Sundays from the dataset.")

    return df_no_sundays