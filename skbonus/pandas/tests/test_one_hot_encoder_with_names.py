"""Test the OneHotEncoderWithNames."""

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from ..preprocessing import OneHotEncoderWithNames


@pytest.fixture
def get_data():
    """Create simple input data."""
    input_data = pd.DataFrame(
        {"A": [1, 2, 3, 2], "B": [0, 0, 0, 1], "C": ["a", "c", "b", "c"]}
    )
    sklearn_ohe_results = OneHotEncoder().fit_transform(input_data)
    skbonus_ohe = OneHotEncoderWithNames()
    skbonus_ohe_results = skbonus_ohe.fit_transform(input_data)
    return skbonus_ohe, sklearn_ohe_results, skbonus_ohe_results


def test_one_hot_encoding(get_data):
    """Test if the values of the OneHotEncoderWithNames matches the ones of scikit-learn's OneHotEncoder."""
    _, sklearn_ohe_results, skbonus_ohe_results = get_data

    assert (sklearn_ohe_results == skbonus_ohe_results.values).all()


def test_column_names(get_data):
    """Test if the columns names are set properly."""
    _, _, skbonus_ohe_results = get_data

    assert skbonus_ohe_results.columns.tolist() == [
        "A_1",
        "A_2",
        "A_3",
        "B_0",
        "B_1",
        "C_a",
        "C_b",
        "C_c",
    ]


def test_assertion_error(get_data):
    """Test if the encoder notices that the column names differ between training and transformation time."""
    df_new = pd.DataFrame(
        {
            "A": [1, 2, 3, 2],
            "C": ["a", "c", "b", "c"],
            "B": [0, 0, 0, 1],
        }
    )

    skbonus_ohe, _, _ = get_data

    with pytest.raises(
        AssertionError,
        match="Column names during fit and transform time should be identical, including the order.",
    ):
        skbonus_ohe.transform(df_new)
