import pandas as pd
import pytest
from src.data import map_into_numeric


@pytest.fixture
def sample_dataframe():
    """
    Fixture providing a sample DataFrame for testing.
    """
    return pd.DataFrame({'store': ['A', 'B', 'C', 'A'],
                         'product_category': ['electronics', 'clothing', 'electronics', 'electronics']})


def test_correct_mapping(sample_dataframe):
    """
    Test if the function correctly maps categorical values to numeric values.
    """
    mappings = {'store': {'A': 1, 'B': 2, 'C': 3}, 'product_category': {'electronics': 1, 'clothing': 2}}
    expected_df = pd.DataFrame({'store': [1, 2, 3, 1], 'product_category': [1, 2, 1, 1]})
    mapped_df = map_into_numeric(sample_dataframe, mappings)
    pd.testing.assert_frame_equal(mapped_df, expected_df)


def test_empty_dataframe():
    """
    Test if the function handles an empty DataFrame.
    """
    mappings = {'store': {'A': 1, 'B': 2}}
    empty_df = pd.DataFrame()
    mapped_df = map_into_numeric(empty_df, mappings)
    assert mapped_df.empty


def test_no_mapping(sample_dataframe):
    """
    Test if the function returns the original DataFrame if no mappings are provided.
    """
    assert map_into_numeric(sample_dataframe, {}) is sample_dataframe


def test_non_dataframe_input():
    """
    Test if the function raises a TypeError when a non-DataFrame input is provided.
    """
    mappings = {'store': {'A': 1, 'B': 2}}
    with pytest.raises(TypeError):
        map_into_numeric("not a dataframe", mappings)


def test_non_dictionary_mappings(sample_dataframe):
    """
    Test if the function raises a TypeError when mappings are not provided as a dictionary.
    """
    with pytest.raises(TypeError):
        map_into_numeric(sample_dataframe, "not a dictionary")
