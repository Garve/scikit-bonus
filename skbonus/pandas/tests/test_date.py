from skbonus.pandas.time import DateFeaturesAdder, Dummifier, PowerTrendAdder
import pandas as pd

dfa = DateFeaturesAdder()
dum = Dummifier(columns=["A", "C"], drop_first=True)
pta = PowerTrendAdder(power=2)

test_df = pd.DataFrame(
    {"A": ["a", "b", "c"], "B": [1, 2, 2], "C": [0, 1, 0]},
    index=[
        pd.Timestamp("1988-08-08"),
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("1950-12-31"),
    ],
)

transformed = dum.fit_transform(test_df)


def test__add_day_of_week():
    assert dfa._add_day_of_week(test_df, use=True).day_of_week.tolist() == [0, 5, 6]


def test__add_day_of_month():
    assert dfa._add_day_of_month(test_df, use=True).day_of_month.tolist() == [8, 1, 31]


def test__add_day_of_year():
    assert dfa._add_day_of_year(test_df, use=True).day_of_year.tolist() == [221, 1, 365]


def test__add_week_of_month():
    assert dfa._add_week_of_month(test_df, use=True).week_of_month.tolist() == [2, 1, 5]


def test__add_week_of_year():
    assert dfa._add_week_of_year(test_df, use=True).week_of_year.tolist() == [
        32,
        52,
        52,
    ]


def test__add_month():
    assert dfa._add_month(test_df, use=True).month.tolist() == [8, 1, 12]


def test__add_year():
    assert dfa._add_year(test_df, use=True).year.tolist() == [1988, 2000, 1950]


# Dummifier
def test_dummifier_transform_column_names():
    assert transformed.columns.tolist() == ["B", "A_b", "A_c", "C_1"]


def test_dummifier_transform_column():
    assert transformed["A_c"].values.tolist() == [0, 0, 1]


# PowerTrendAdder
def test_powertrendadder_tranform():
    sorted_test_df = test_df.sort_index()
    test_df_with_trend = pta.fit_transform(sorted_test_df)
    assert "trend" in test_df_with_trend.columns
    assert test_df_with_trend.trend.tolist() == [0.0, 1.0, 1.698054714750539]
