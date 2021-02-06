import pandas as pd
import pytest

from skbonus.pandas.time import DateFeaturesAdder, SpecialDatesAdder, PowerTrendAdder

dfa = DateFeaturesAdder(
    day_of_week=True,
    day_of_month=True,
    day_of_year=True,
    week_of_month=True,
    week_of_year=True,
    month=True,
    year=True,
)

sda = SpecialDatesAdder(
    "black_friday_2018",
    ["2018-11-23"],
    window=15,
    center=True,
    win_type="gaussian",
    std=1,
)

pta = PowerTrendAdder()

non_continuous_input = pd.DataFrame(
    {"A": ["a", "b", "c"], "B": [1, 2, 2], "C": [0, 1, 0]},
    index=[
        pd.Timestamp("1988-08-08"),
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("1950-12-31"),
    ],
)

continuous_input = pd.DataFrame(
    {"data": range(60)}, index=pd.date_range(start="2018-11-01", periods=60)
)


def test_datefeaturesadder__add_day_of_week():
    assert dfa._add_day_of_week(non_continuous_input).day_of_week.tolist() == [0, 5, 6]


def test_datefeaturesadder__add_day_of_month():
    assert dfa._add_day_of_month(non_continuous_input).day_of_month.tolist() == [
        8,
        1,
        31,
    ]


def test_datefeaturesadder__add_day_of_year():
    assert dfa._add_day_of_year(non_continuous_input).day_of_year.tolist() == [
        221,
        1,
        365,
    ]


def test_datefeaturesadder__add_week_of_month():
    assert dfa._add_week_of_month(non_continuous_input).week_of_month.tolist() == [
        2,
        1,
        5,
    ]


def test_datefeaturesadder__add_week_of_year():
    assert dfa._add_week_of_year(non_continuous_input).week_of_year.tolist() == [
        32,
        52,
        52,
    ]


def test_datefeaturesadder__add_month():
    assert dfa._add_month(non_continuous_input).month.tolist() == [8, 1, 12]


def test_datefeaturesadder__add_year():
    assert dfa._add_year(non_continuous_input).year.tolist() == [1988, 2000, 1950]


def test_special_day_adder_fit_transform():
    transformed = sda.fit_transform(continuous_input)
    assert transformed.loc["2018-11-23", "black_friday_2018"] == 1.0
    assert 0.6 < transformed.loc["2018-11-22", "black_friday_2018"] < 0.61
    assert 0.6 < transformed.loc["2018-11-24", "black_friday_2018"] < 0.61
    assert 0.13 < transformed.loc["2018-11-21", "black_friday_2018"] < 0.14
    assert 0.13 < transformed.loc["2018-11-25", "black_friday_2018"] < 0.14


def test_powertrendadder_exception():
    with pytest.raises(ValueError):
        pta.fit(non_continuous_input)


def test_powertrendadder_fit_transform():
    assert pta.fit_transform(continuous_input).trend.tolist() == list(range(60))
