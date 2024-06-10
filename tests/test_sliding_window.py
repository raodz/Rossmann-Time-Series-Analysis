import pandas as pd
import numpy as np
import pytest
from src.data import sliding_window


@pytest.fixture
def sample_dataframe():
    """
    Fixture providing a sample DataFrame for testing.
    """
    return pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         'B': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         'Sales': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]})


def test_sliding_window(sample_dataframe):
    """
    Test if the function generates correct sliding windows for input features (X) and corresponding target values (y).
    """
    X_expected = {'A': [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]],
                  'B': [[11, 12, 13, 14, 15, 16, 17], [12, 13, 14, 15, 16, 17, 18], [13, 14, 15, 16, 17, 18, 19]],
                  'Sales': [[21, 22, 23, 24, 25, 26, 27], [22, 23, 24, 25, 26, 27, 28], [23, 24, 25, 26, 27, 28, 29]]}
    y_expected = [28, 29, 30]

    X_actual, y_actual = sliding_window(sample_dataframe)
    # X_actual contains pd.Series objects that need to be converted into lists to be compared
    X_actual = {key: [window.tolist() for window in value] for key, value in X_actual.items()}

    assert X_actual['A'] == X_expected['A']
    assert X_actual['B'] == X_expected['B']
    assert X_actual['Sales'] == X_expected['Sales']
    assert y_actual == y_expected


def test_non_dataframe_input():
    """
    Test if the function raises a AttributeError when an object without columns input is provided.
    """
    with pytest.raises(AttributeError):
        sliding_window("the object with no columns")


def test_non_existing_target_column(sample_dataframe):
    """
    Test if the function raises a KeyError when the target column does not exist in the DataFrame.
    """
    with pytest.raises(KeyError):
        sliding_window(sample_dataframe, 'non_existing_column')


def test_sliding_window_with_too_short_df():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [11, 12, 13, 14, 15, 16], 'Sales': [21, 22, 23, 24, 25, 26]})

    with pytest.raises(ValueError, match="The length of the DataFrame must be greater than the window size."):
        sliding_window(df, 'Sales')


def test_sliding_window_custom_y_col():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'B': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                       'C': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]})
    X_actual, y_actual = sliding_window(df, 'C')
    # X_actual contains pd.Series objects that need to be converted into lists to be compared
    X_actual = {key: [window.tolist() for window in value] for key, value in X_actual.items()}
    expected_X = {
        'A': [[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]],
        'B': [[11, 12, 13, 14, 15, 16, 17], [12, 13, 14, 15, 16, 17, 18], [13, 14, 15, 16, 17, 18, 19]],
        'C': [[21, 22, 23, 24, 25, 26, 27], [22, 23, 24, 25, 26, 27, 28], [23, 24, 25, 26, 27, 28, 29]]
    }
    expected_y = [28, 29, 30]

    assert X_actual == expected_X
    assert y_actual == expected_y
