import pandas as pd
import pytest
from src.data import get_X_y


def sample_dataframe():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'Sales': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    })


def test_get_X_y_basic():
    df = sample_dataframe()
    X, y = get_X_y(df, 'Sales')
    expected_X = pd.DataFrame({
        'A0': [1, 2, 3],
        'A1': [2, 3, 4],
        'A2': [3, 4, 5],
        'A3': [4, 5, 6],
        'A4': [5, 6, 7],
        'A5': [6, 7, 8],
        'A6': [7, 8, 9],
        'B0': [11, 12, 13],
        'B1': [12, 13, 14],
        'B2': [13, 14, 15],
        'B3': [14, 15, 16],
        'B4': [15, 16, 17],
        'B5': [16, 17, 18],
        'B6': [17, 18, 19],
        'Sales0': [21, 22, 23],
        'Sales1': [22, 23, 24],
        'Sales2': [23, 24, 25],
        'Sales3': [24, 25, 26],
        'Sales4': [25, 26, 27],
        'Sales5': [26, 27, 28],
        'Sales6': [27, 28, 29]
    }, index=df.index[7:])

    expected_y = pd.Series([28, 29, 30], index=df.index[7:])

    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_series_equal(y, expected_y)


def test_get_X_y_custom_y_col():
    df = sample_dataframe()
    df.rename(columns={'Sales': 'Target'}, inplace=True)
    X, y = get_X_y(df, 'Target')
    expected_X = pd.DataFrame({
        'A0': [1, 2, 3],
        'A1': [2, 3, 4],
        'A2': [3, 4, 5],
        'A3': [4, 5, 6],
        'A4': [5, 6, 7],
        'A5': [6, 7, 8],
        'A6': [7, 8, 9],
        'B0': [11, 12, 13],
        'B1': [12, 13, 14],
        'B2': [13, 14, 15],
        'B3': [14, 15, 16],
        'B4': [15, 16, 17],
        'B5': [16, 17, 18],
        'B6': [17, 18, 19],
        'Target0': [21, 22, 23],
        'Target1': [22, 23, 24],
        'Target2': [23, 24, 25],
        'Target3': [24, 25, 26],
        'Target4': [25, 26, 27],
        'Target5': [26, 27, 28],
        'Target6': [27, 28, 29]
    }, index=df.index[7:])

    expected_y = pd.Series([28, 29, 30], index=df.index[7:])

    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_series_equal(y, expected_y)


def test_get_X_y_with_too_short_df():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [11, 12, 13, 14, 15, 16], 'Sales': [21, 22, 23, 24, 25, 26]})

    with pytest.raises(ValueError, match="The length of the DataFrame must be greater than the window size."):
        get_X_y(df, 'Sales')
