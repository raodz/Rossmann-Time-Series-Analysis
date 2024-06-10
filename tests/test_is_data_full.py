import pandas as pd
import pytest
from src.data import is_data_full


@pytest.fixture
def sample_sales_dataframe():
    """
    Fixture providing a sample DataFrame for testing.
    """
    return pd.DataFrame({'sales': [100, 150, 200]}, index=('2024-01-01', '2024-01-02', '2024-01-03'))


def test_sales_data_full(sample_sales_dataframe):
    """
    Test if the function returns True when sales DataFrame spans consecutive days without any missing dates.
    """
    assert is_data_full(sample_sales_dataframe) == True


def test_sales_data_not_full():
    """
    Test if the function returns False when sales DataFrame does not span consecutive days.
    """
    df = pd.DataFrame({'sales': [100, 200]}, index=('2024-01-01', '2024-01-03'))
    assert is_data_full(df) == False


def test_empty_sales_dataframe():
    """
    Test if the function raises a ValueError when an empty sales DataFrame is provided.
    """
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        is_data_full(empty_df)


def test_non_dataframe_sales_input():
    """
    Test if the function raises a TypeError when a non-DataFrame input is provided for sales data.
    """
    with pytest.raises(TypeError):
        is_data_full("not a dataframe")


def test_invalid_sales_date_format():
    """
    Test if the function raises a ValueError when the sales DataFrame index is not in the expected '%Y-%m-%d' format.
    """
    df = pd.DataFrame({'sales': [100, 150, 200]}, index=['2024/01/01', '2024/01/02', '2024/01/03'])
    with pytest.raises(ValueError):
        is_data_full(df)


def test_single_day_sales_dataframe():
    """
    Test if the function returns True when sales DataFrame spans only one day.
    """
    df = pd.DataFrame({'sales': [100]}, index=['2024-01-01'])
    assert is_data_full(df) == True


def test_multi_year_sales_dataframe():
    """
    Test if the function returns True when sales DataFrame spans consecutive days over multiple years.
    """
    df = pd.DataFrame({'sales': [100, 150]}, index=['2023-12-31', '2024-01-01'])
    assert is_data_full(df) == True
