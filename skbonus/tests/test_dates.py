from skbonus.pandas.dates import DateFeaturesAdder
import pandas as pd

d = DateFeaturesAdder()
test_df = pd.DataFrame(
    {"A": [1, 2, 3]},
    index=[
        pd.Timestamp("1988-08-08"),
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("1950-12-31"),
    ],
)


def test__add_day_of_week():
    assert d._add_day_of_week(test_df, use=True).day_of_week.tolist() == [0, 5, 6]


def test__add_day_of_month():
    assert d._add_day_of_month(test_df, use=True).day_of_month.tolist() == [8, 1, 31]


def test__add_day_of_year():
    assert d._add_day_of_year(test_df, use=True).day_of_year.tolist() == [221, 1, 365]


def test__add_week_of_month():
    assert d._add_week_of_month(test_df, use=True).week_of_month.tolist() == [2, 1, 5]


def test__add_week_of_year():
    assert d._add_week_of_year(test_df, use=True).week_of_year.tolist() == [32, 52, 52]


def test__add_month():
    assert d._add_month(test_df, use=True).month.tolist() == [8, 1, 12]


def test__add_year():
    assert d._add_year(test_df, use=True).year.tolist() == [1988, 2000, 1950]
