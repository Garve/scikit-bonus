"""Deal with time features in dataframes."""

import numpy as np
import pandas as pd


def explode_date(
    df: pd.DataFrame,
    start_column: str,
    end_column: str,
    result_column: str = "Date",
    frequency: str = "d",
    drop: bool = True,
) -> pd.DataFrame:
    """
    Transform a pandas dataframe with columns (*, start_date, end_date) into a longer format with columns (*, date).

    This is useful if you deal with datasets that contain special time periods per row, but you need a single date per row.
    See the examples for more details.

    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe with a column containing starts dates and end dates.

    start_column : str
        Start date of the period.

    end_column : str
        End date of the period.

    result_column : str, default="Date"
        Name of the new output date column.

    frequency : str, default="d" (for day)
        A pandas time frequency. Can take values like "d" for day or "m" for month. A full list can
        be found on https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        If None, the transformer tries to infer it during fit time.

    drop : bool, default=True
        Whether to drop the `start_column` and `end_column` in the output.

    Returns
    -------
    pd.DataFrame
        A longer dataframe with one date per row.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    "Data": ["a", "b", "c"],
    ...    "Start": pd.date_range("2020-01-01", periods=3),
    ...    "End": pd.date_range("2020-01-03", periods=3)
    ... })
    >>> df
      Data      Start        End
    0    a 2020-01-01 2020-01-03
    1    b 2020-01-02 2020-01-04
    2    c 2020-01-03 2020-01-05

    >>> explode_date(df, start_column="Start", end_column="End", result_column="output_date", frequency="d")
      Data output_date
    0    a  2020-01-01
    0    a  2020-01-02
    0    a  2020-01-03
    1    b  2020-01-02
    1    b  2020-01-03
    1    b  2020-01-04
    2    c  2020-01-03
    2    c  2020-01-04
    2    c  2020-01-05
    """
    return (
        df.assign(
            **{
                result_column: lambda df: df.apply(
                    lambda row: pd.date_range(
                        start=row[start_column], end=row[end_column], freq=frequency
                    ),
                    axis=1,
                )
            }
        )
        .explode(result_column)
        .drop(columns=drop * [start_column, end_column])
    )


def add_date_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Enrich a pandas dataframes with a new column indicating if there is a special date.

    This new column will contain a one for each date specified in the `dates` keyword, zero otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a DateTime index.

    kwargs : List[str]*
        As many inputs as you want of the form date_name=[date_1, date_2, ...], i.e. christmas=['2020-12-24'].
        See the example below for more information.

    Returns
    -------
    pd.DataFrame
        A dataframe with date indicator columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": range(7)}, index=pd.date_range(start="2019-12-29", periods=7))
    >>> add_date_indicators(
    ...     df,
    ...     around_new_year_2020=["2019-12-31", "2020-01-01", "2020-01-02"],
    ...     other_date_1=["2019-12-29"],
    ...     other_date_2=["2018-01-01"]
    ... )
                A  around_new_year_2020  other_date_1  other_date_2
    2019-12-29  0                     0             1             0
    2019-12-30  1                     0             0             0
    2019-12-31  2                     1             0             0
    2020-01-01  3                     1             0             0
    2020-01-02  4                     1             0             0
    2020-01-03  5                     0             0             0
    2020-01-04  6                     0             0             0
    """
    return df.assign(
        **{name: df.index.isin(dates).astype(int) for name, dates in kwargs.items()}
    )


def add_time_features(
    df: pd.DataFrame,
    second: bool = False,
    minute: bool = False,
    hour: bool = False,
    day_of_week: bool = False,
    day_of_month: bool = False,
    day_of_year: bool = False,
    week_of_month: bool = False,
    week_of_year: bool = False,
    month: bool = False,
    year: bool = False,
) -> pd.DataFrame:
    """
    Enrich pandas dataframes with new columns which are easy derivations from its DatetimeIndex, such as the day of week or the month.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe with a DateTime index.

    second : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    minute : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    hour : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    day_of_week : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    day_of_month : bool, default=False
        Whether to extract the day of month from the index and add it as a new column.

    day_of_year : bool, default=False
        Whether to extract the day of year from the index and add it as a new column.

    week_of_month : bool, default=False
        Whether to extract the week of month from the index and add it as a new column.

    week_of_year : bool, default=False
        Whether to extract the week of year from the index and add it as a new column.

    month : bool, default=False
        Whether to extract the month from the index and add it as a new column.

    year : bool, default=False
        Whether to extract the year from the index and add it as a new column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"A": ["a", "b", "c"]},
    ...     index=[
    ...         pd.Timestamp("1988-08-08"),
    ...         pd.Timestamp("2000-01-01"),
    ...         pd.Timestamp("1950-12-31"),
    ...     ])
    >>> add_time_features(df, day_of_month=True, month=True, year=True)
                A  day_of_month  month  year
    1988-08-08  a             8      8  1988
    2000-01-01  b             1      1  2000
    1950-12-31  c            31     12  1950
    """

    def _add_second(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(second=df.index.second) if second else df

    def _add_minute(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(minute=df.index.minute) if minute else df

    def _add_hour(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(hour=df.index.hour) if hour else df

    def _add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(day_of_week=df.index.weekday + 1) if day_of_week else df

    def _add_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(day_of_month=df.index.day) if day_of_month else df

    def _add_day_of_year(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(day_of_year=df.index.dayofyear) if day_of_year else df

    def _add_week_of_month(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.assign(week_of_month=np.ceil(df.index.day / 7).astype(int))
            if week_of_month
            else df
        )

    def _add_week_of_year(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.assign(week_of_year=df.index.isocalendar().week) if week_of_year else df
        )

    def _add_month(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(month=df.index.month) if month else df

    def _add_year(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(year=df.index.year) if year else df

    return (
        df.pipe(_add_second)
        .pipe(_add_minute)
        .pipe(_add_hour)
        .pipe(_add_day_of_week)
        .pipe(_add_day_of_month)
        .pipe(_add_day_of_year)
        .pipe(_add_week_of_month)
        .pipe(_add_week_of_year)
        .pipe(_add_month)
        .pipe(_add_year)
    )
