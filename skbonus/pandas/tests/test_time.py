import pandas as pd

from skbonus.pandas.time import DateFeaturesAdder, PowerTrendAdder

dfa = DateFeaturesAdder(
    day_of_week=True,
    day_of_month=True,
    day_of_year=True,
    week_of_month=True,
    week_of_year=True,
    month=True,
    year=True,
    special_dates={"new_year_2000": [pd.Timestamp("2000-01-01")]},
)
pta = PowerTrendAdder(power=2)

test_df = pd.DataFrame(
    {"A": ["a", "b", "c"], "B": [1, 2, 2], "C": [0, 1, 0]},
    index=[
        pd.Timestamp("1988-08-08"),
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("1950-12-31"),
    ],
)


def test__add_day_of_week():
    assert dfa._add_day_of_week(test_df).day_of_week.tolist() == [0, 5, 6]


def test__add_day_of_month():
    assert dfa._add_day_of_month(test_df).day_of_month.tolist() == [8, 1, 31]


def test__add_day_of_year():
    assert dfa._add_day_of_year(test_df).day_of_year.tolist() == [221, 1, 365]


def test__add_week_of_month():
    assert dfa._add_week_of_month(test_df).week_of_month.tolist() == [2, 1, 5]


def test__add_week_of_year():
    assert dfa._add_week_of_year(test_df).week_of_year.tolist() == [
        32,
        52,
        52,
    ]


def test__add_month():
    assert dfa._add_month(test_df).month.tolist() == [8, 1, 12]


def test__add_year():
    assert dfa._add_year(test_df).year.tolist() == [1988, 2000, 1950]


def test__add_special_dates():
    assert dfa._add_special_dates(test_df).new_year_2000.tolist() == [0, 1, 0]
