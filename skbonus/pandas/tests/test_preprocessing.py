"""Test preprocessing steps."""

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from skbonus.pandas.preprocessing import OneHotEncoderWithNames

df = pd.DataFrame({"A": [1, 2, 3, 2], "B": [0, 0, 0, 1], "C": ["a", "c", "b", "c"]})
ohe_sklearn = OneHotEncoder().fit_transform(df)

ohe_with_names = OneHotEncoderWithNames()
ohe_skbonus = ohe_with_names.fit_transform(df)


def test_one_hot_encoding():
    """Test if the values of the OneHotEncoderWithNames matches the ones of scikit-learn's OneHotEncoder."""
    assert (ohe_sklearn == ohe_skbonus.values).all()


def test_column_names():
    """Test if the columns names are set properly."""
    assert ohe_skbonus.columns.tolist() == [
        "A_1",
        "A_2",
        "A_3",
        "B_0",
        "B_1",
        "C_a",
        "C_b",
        "C_c",
    ]


def test_assertion_error():
    """Test if the encoder notices that the column names differ between training and transformation time."""
    df_new = pd.DataFrame(
        {
            "A": [1, 2, 3, 2],
            "C": ["a", "c", "b", "c"],
            "B": [0, 0, 0, 1],
        }
    )

    with pytest.raises(
        AssertionError,
        match="Column names during fit and transform time should be identical, including the order.",
    ):
        ohe_with_names.transform(df_new)
