import numpy as np
import pandas as pd
import pytest

from skbonus.pandas.time import (
    SimpleTimeFeatures,
    SpecialDayBumps,
    PowerTrend,
    CyclicalEncoder,
)

from skbonus.exceptions import NoFrequencyError

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

sda = SpecialDayBumps(
    "black_friday_2018",
    ["2018-11-23"],
    window=15,
    win_type="general_gaussian",
    p=1,
    sig=1,
)

pta = PowerTrend()

ce = CyclicalEncoder()

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

minutes = pd.DataFrame({"minute": range(60)})

sda_transformed = sda.fit_transform(continuous_input)
ce_transformed = ce.fit_transform(minutes)


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
    assert function(non_continuous_input)[column_name].tolist() == result


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
    np.testing.assert_almost_equal(
        sda_transformed.loc[date, "black_friday_2018"], value
    )


def test_powertrendadder_exception():
    with pytest.raises(NoFrequencyError):
        pta.fit(non_continuous_input)


def test_powertrendadder_fit_transform():
    assert pta.fit_transform(continuous_input).trend.tolist() == list(range(60))


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
    np.testing.assert_almost_equal(ce_transformed.minute_cos.loc[index], value)
