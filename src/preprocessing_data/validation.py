import datetime as dt
import pandas as pd


def is_data_full(df: pd.DataFrame) -> bool:
    """Check if the given DataFrame spans consecutive days without any missing dates."""
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