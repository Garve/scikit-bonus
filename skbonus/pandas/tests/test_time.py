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


def test_simple_time_features_fit_transform():
    """Test SimpleTimeFeatures' fit and transform."""
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

    assert (
        (
            dfa.fit_transform(non_continuous_input)
            == pd.DataFrame(
                {
                    "A": ["a", "b", "c"],
                    "B": [1, 2, 2],
                    "C": [0, 1, 0],
                    "second": [12, 5, 0],
                    "minute": [12, 6, 0],
                    "hour": [11, 7, 0],
                    "day_of_week": [1, 6, 7],
                    "day_of_month": [8, 1, 31],
                    "day_of_year": [221, 1, 365],
                    "week_of_month": [2, 1, 5],
                    "week_of_year": [32, 52, 52],
                    "month": [8, 1, 12],
                    "year": [1988, 2000, 1950],
                },
                index=non_continuous_input.index,
            )
        )
        .all()
        .all()
    )


def test_special_day_bumps_fit_transform():
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
        sda_transformed.loc["2018-11-21":"2018-11-25", "black_friday_2018"].values,
        [0.1353352832366127, 6.065307e-01, 1.0, 6.065307e-01, 0.1353352832366127],
    )


def test_special_day_bumps_no_freq():
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
        sda_transformed.loc["2018-11-21":"2018-11-25", "black_friday_2018"].values,
        [0.1353352832366127, 6.065307e-01, 1.0, 6.065307e-01, 0.1353352832366127],
    )


def test_special_day_bumps_no_freq_error():
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


def test_power_trend_adder_fit_transform():
    """Test the PowerTrendAdder."""
    pta = PowerTrend(frequency="d", origin_date="2018-11-01")
    assert pta.fit_transform(continuous_input).trend.tolist() == list(range(60))


def test_power_trend_adder_fit_transform_defaults():
    """Test the PowerTrendAdder without provided frequency and origin_date."""
    pta = PowerTrend()
    assert pta.fit_transform(continuous_input).trend.tolist() == list(range(60))
    assert pta.freq_ == "D"
    assert pta.origin_ == pd.Timestamp("2018-11-01", freq="D")


def test_power_trend_adder_fit_transform_defaults_error():
    """Test the PowerTrendAdder without provided frequency and origin_date, and without the possibility to extract it during fit time."""
    pta = PowerTrend()
    with pytest.raises(ValueError):
        pta.fit(non_continuous_input)


def test_cyclical_encoder():
    """Test the CyclicalEncoder."""
    ce = CyclicalEncoder()
    minutes = pd.DataFrame({"minute": range(60)})
    ce_transformed = ce.fit_transform(minutes)

    np.testing.assert_almost_equal(
        ce_transformed.minute_cos.values[:7],
        np.array(
            [
                1.0000000e00,
                9.9452190e-01,
                9.7814760e-01,
                9.5105652e-01,
                9.1354546e-01,
                8.6602540e-01,
                8.0901699e-01,
            ]
        ),
    )


def test_cyclical_encoder_additional_cycles():
    """Test if the additional_cyclces."""
    ce = CyclicalEncoder(additional_cycles={"TEST": {"min": 0, "max": 99}})
    test = pd.DataFrame({"TEST": range(10)})
    assert ce.fit_transform(test).columns.tolist() == ["TEST", "TEST_cos", "TEST_sin"]
