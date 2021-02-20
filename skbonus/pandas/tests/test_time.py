"""Test time classes."""

import numpy as np
import pandas as pd
import pytest

from ..time import (
    CyclicalEncoder,
    PowerTrend,
    SimpleTimeFeatures,
    SpecialDayBumps,
)

dfa = SimpleTimeFeatures(
    second=True,
    minute=True,
    hour=True,
    day_of_week=True,
    day_of_month=True,
    day_of_year=True,
    week_of_month=True,
    week_of_year=True,
    month=True,
    year=True,
)

non_continuous_input = pd.DataFrame(
    {"A": ["a", "b", "c"], "B": [1, 2, 2], "C": [0, 1, 0]},
    index=[
        pd.Timestamp("1988-08-08 11:12:12"),
        pd.Timestamp("2000-01-01 07:06:05"),
        pd.Timestamp("1950-12-31"),
    ],
)

continuous_input = pd.DataFrame(
    {"data": range(60)}, index=pd.date_range(start="2018-11-01", periods=60)
)


@pytest.mark.parametrize(
    "function, column_name, result",
    [
        (dfa._add_second, "second", [12, 5, 0]),
        (dfa._add_minute, "minute", [12, 6, 0]),
        (dfa._add_hour, "hour", [11, 7, 0]),
        (dfa._add_day_of_week, "day_of_week", [1, 6, 7]),
        (dfa._add_day_of_month, "day_of_month", [8, 1, 31]),
        (dfa._add_day_of_year, "day_of_year", [221, 1, 365]),
        (dfa._add_week_of_month, "week_of_month", [2, 1, 5]),
        (dfa._add_week_of_year, "week_of_year", [32, 52, 52]),
        (dfa._add_month, "month", [8, 1, 12]),
        (dfa._add_year, "year", [1988, 2000, 1950]),
    ],
)
def test_timefeaturesadder(function, column_name, result):
    """Test SimpleTimeFeatures' methods."""
    assert function(non_continuous_input)[column_name].tolist() == result


def test_simple_time_features_fit():
    """Test if fit / transform in SimpleTimeFeatuters work."""
    assert dfa.fit_transform(non_continuous_input).columns.tolist() == [
        "A",
        "B",
        "C",
        "second",
        "minute",
        "hour",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week_of_month",
        "week_of_year",
        "month",
        "year",
    ]


@pytest.mark.parametrize(
    "date, value",
    [
        ("2018-11-23", 1.0),
        ("2018-11-22", 6.065307e-01),
        ("2018-11-24", 6.065307e-01),
        ("2018-11-21", 0.1353352832366127),
        ("2018-11-25", 0.1353352832366127),
    ],
)
def test_specialeventsadder(date, value):
    """Test the SpecialDayBumps."""
    sda = SpecialDayBumps(
        name="black_friday_2018",
        dates=["2018-11-23"],
        frequency="d",
        window=15,
        p=1,
        sig=1,
    )

    sda_transformed = sda.fit_transform(continuous_input)

    np.testing.assert_almost_equal(
        sda_transformed.loc[date, "black_friday_2018"], value
    )


@pytest.mark.parametrize(
    "date, value",
    [
        ("2018-11-23", 1.0),
        ("2018-11-22", 6.065307e-01),
        ("2018-11-24", 6.065307e-01),
        ("2018-11-21", 0.1353352832366127),
        ("2018-11-25", 0.1353352832366127),
    ],
)
def test_specialeventsadder_no_freq(date, value):
    """Test the SpecialDayBumps with the frequency inferred."""
    sda = SpecialDayBumps(
        name="black_friday_2018",
        dates=["2018-11-23"],
        window=15,
        p=1,
        sig=1,
    )

    sda_transformed = sda.fit_transform(continuous_input)

    np.testing.assert_almost_equal(
        sda_transformed.loc[date, "black_friday_2018"], value
    )


def test_specialeventsadder_no_freq_error():
    """Test the SpecialDayBumps without frequency provided, and where it cannot be inferred during fit time."""
    sda = SpecialDayBumps(
        name="black_friday_2018",
        dates=["2018-11-23"],
        window=15,
        p=1,
        sig=1,
    )

    with pytest.raises(ValueError):
        sda.fit(non_continuous_input)


def test_powertrendadder_fit_transform():
    """Test the PowerTrendAdder."""
    pta = PowerTrend(frequency="d", origin_date="2018-11-01")
    assert pta.fit_transform(continuous_input).trend.tolist() == list(range(60))


def test_powertrendadder_fit_transform_defaults():
    """Test the PowerTrendAdder without provided frequency and origin_date."""
    pta = PowerTrend()
    assert pta.fit_transform(continuous_input).trend.tolist() == list(range(60))
    assert pta.freq_ == "D"
    assert pta.origin_ == pd.Timestamp("2018-11-01", freq="D")


def test_powertrendadder_fit_transform_defaults_error():
    """Test the PowerTrendAdder without provided frequency and origin_date, and without the possibility to extract it during fit time."""
    pta = PowerTrend()
    with pytest.raises(ValueError):
        pta.fit(non_continuous_input)


@pytest.mark.parametrize(
    "index, value",
    [
        (0, 1.0),
        (1, 9.945219e-01),
        (2, 9.781476e-01),
        (3, 9.510565e-01),
        (4, 9.135455e-01),
    ],
)
def test_cyclicalencoder(index, value):
    """Test the CyclicalEncoder."""
    ce = CyclicalEncoder()
    minutes = pd.DataFrame({"minute": range(60)})
    ce_transformed = ce.fit_transform(minutes)

    np.testing.assert_almost_equal(ce_transformed.minute_cos.loc[index], value)


def test_cyclicalencoder_additional_cycles():
    """Test if the additional_cyclces."""
    ce = CyclicalEncoder(additional_cycles={"TEST": {"min": 0, "max": 99}})
    test = pd.DataFrame({"TEST": range(10)})
    assert ce.fit_transform(test).columns.tolist() == ["TEST", "TEST_cos", "TEST_sin"]
