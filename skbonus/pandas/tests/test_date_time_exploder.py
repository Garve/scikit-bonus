"""Test the DateTimeExploder."""

import pandas as pd
import pytest

from ..preprocessing import DateTimeExploder


@pytest.fixture
def get_data():
    """Create simple input data."""
    input_data = pd.DataFrame(
        {
            "Data": ["a", "b", "c"],
            "Start": pd.date_range("2020-01-01", periods=3),
            "End": pd.date_range("2020-01-03", periods=3),
        }
    )

    return input_data


def test_fit_transform(get_data):
    """Test if fit_transform with a non-existent column name fails."""
    input_data = get_data
    d = DateTimeExploder(
        name="output_date", start_column="___Start___", end_column="End", frequency="d"
    )

    with pytest.raises(KeyError):
        d.fit_transform(input_data)
