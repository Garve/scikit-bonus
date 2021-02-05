import pandas as pd

from skbonus.pandas.time import DateFeaturesAdder, SpecialDatesAdder

dfa = DateFeaturesAdder(
    day_of_week=True,
    day_of_month=True,
    day_of_year=True,
    week_of_month=True,
    week_of_year=True,
    month=True,
    year=True,
)

dfa_input = pd.DataFrame(
    {"A": ["a", "b", "c"], "B": [1, 2, 2], "C": [0, 1, 0]},
    index=[
        pd.Timestamp("1988-08-08"),
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("1950-12-31"),
    ],
)

sda = SpecialDatesAdder(
    "black_friday_2018",
    ["2018-11-23"],
    window=15,
    center=True,
    win_type="gaussian",
    std=1,
)
sda_input = pd.DataFrame(
    {"data": range(60)}, index=pd.date_range(start="2018-11-01", periods=60)
)


def test__add_day_of_week():
    assert dfa._add_day_of_week(dfa_input).day_of_week.tolist() == [0, 5, 6]


def test__add_day_of_month():
    assert dfa._add_day_of_month(dfa_input).day_of_month.tolist() == [8, 1, 31]


def test__add_day_of_year():
    assert dfa._add_day_of_year(dfa_input).day_of_year.tolist() == [221, 1, 365]


def test__add_week_of_month():
    assert dfa._add_week_of_month(dfa_input).week_of_month.tolist() == [2, 1, 5]


def test__add_week_of_year():
    assert dfa._add_week_of_year(dfa_input).week_of_year.tolist() == [
        32,
        52,
        52,
    ]


def test__add_month():
    assert dfa._add_month(dfa_input).month.tolist() == [8, 1, 12]


def test__add_year():
    assert dfa._add_year(dfa_input).year.tolist() == [1988, 2000, 1950]


def test__special_day_adder_fit_transform():
    transformed = sda.fit_transform(sda_input)
    assert transformed.loc["2018-11-23", "black_friday_2018"] == 1.0
    assert 0.6 < transformed.loc["2018-11-22", "black_friday_2018"] < 0.61
    assert 0.6 < transformed.loc["2018-11-24", "black_friday_2018"] < 0.61
    assert 0.13 < transformed.loc["2018-11-21", "black_friday_2018"] < 0.14
    assert 0.13 < transformed.loc["2018-11-25", "black_friday_2018"] < 0.14
