import pandas as pd
from unittest.mock import patch
import pytest
from src.data import preprocess_data

NUM_DAYS_IN_WEEK = 7


def sample_dataframe():
    """
    Generate a sample DataFrame for testing preprocess_data function.
    """
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'Sales': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'StateHoliday': ['0', '0', '0', 'a', 'b', 'c', '0', 'a', 'b', 'c',
                         '0', '0', '0', 'a', 'b', 'c', '0', 'a', 'b', 'c',
                         '0', '0', '0', 'a', 'b', 'c', '0', 'a', 'b', 'c',
                         '0', '0', '0', 'a', 'b', 'c', '0', 'a', 'b', 'c'],
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                 '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10',
                 '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15',
                 '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20',
                 '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                 '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10',
                 '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15',
                 '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-20'],
        'Store': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    })


def test_preprocess_data():
    """
    Test preprocess_data function.
    """
    with patch('src.data.load_data', return_value=sample_dataframe()), \
            patch('src.data.is_data_full', return_value=True):
        file_name = 'dummy.csv'
        y_col = 'Sales'
        prc_samples_for_test = 0.2

        X_train, y_train, X_test, y_test = preprocess_data(file_name, y_col, prc_samples_for_test)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)

        assert len(X_train) == 13
        assert len(y_train) == 13
        assert len(X_test) == 1
        assert len(y_test) == 1

        assert 'StateHoliday' not in X_train.columns
        assert 'StateHoliday' not in X_test.columns


def test_preprocess_data_file_not_found():
    """
    Test preprocess_data function with a non-existent file.
    """
    with pytest.raises(FileNotFoundError):
        preprocess_data('non_existent_file.csv')


def test_preprocess_data_assertion_error():
    """
    Test preprocess_data function with data not being full.
    """
    with patch('src.data.load_data', return_value=sample_dataframe()), \
            patch('src.data.is_data_full', return_value=False):
        with pytest.raises(AssertionError):
            preprocess_data('dummy.csv')
