from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class SimpleTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Enrich pandas dataframes with new columns which are easy derivations from its DatetimeIndex, such as the day of week or the month.

    Parameters
    ----------
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
    >>> SimpleTimeFeatures(day_of_month=True, month=True, year=True).fit_transform(df)
                A  day_of_month  month  year
    1988-08-08  a             8      8  1988
    2000-01-01  b             1      1  2000
    1950-12-31  c            31     12  1950
    """

    def __init__(
        self,
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
    ) -> None:
        """Initialize."""
        self.second = second
        self.minute = minute
        self.hour = hour
        self.day_of_week = day_of_week
        self.day_of_month = day_of_month
        self.day_of_year = day_of_year
        self.week_of_month = week_of_month
        self.week_of_year = week_of_year
        self.month = month
        self.year = year

    def _add_second(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(second=lambda df: df.index.second) if self.second else X

    def _add_minute(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(minute=lambda df: df.index.minute) if self.minute else X

    def _add_hour(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(hour=lambda df: df.index.hour) if self.hour else X

    def _add_day_of_week(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            X.assign(day_of_week=lambda df: df.index.weekday + 1)
            if self.day_of_week
            else X
        )

    def _add_day_of_month(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            X.assign(day_of_month=lambda df: df.index.day) if self.day_of_month else X
        )

    def _add_day_of_year(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            X.assign(day_of_year=lambda df: df.index.dayofyear)
            if self.day_of_year
            else X
        )

    def _add_week_of_month(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            X.assign(week_of_month=lambda df: np.ceil(df.index.day / 7).astype(int))
            if self.week_of_month
            else X
        )

    def _add_week_of_year(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            X.assign(week_of_year=lambda df: df.index.isocalendar().week)
            if self.week_of_year
            else X
        )

    def _add_month(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(month=lambda df: df.index.month) if self.month else X

    def _add_year(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(year=lambda df: df.index.year) if self.year else X

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SimpleTimeFeatures":
        """
        Fit the estimator.

        In this special case, nothing is done.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        SimpleTimeFeatures
            Fitted transformer.
        """
        self._check_n_features(X, reset=True)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Insert all chosen time features as new columns into the dataframe and output it.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas dataframe with a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            The input dataframe with additional time feature columns.
        """
        check_is_fitted(self)
        check_array(X, dtype=None)
        self._check_n_features(X, reset=False)

        res = (
            X.pipe(self._add_second)
            .pipe(self._add_minute)
            .pipe(self._add_hour)
            .pipe(self._add_day_of_week)
            .pipe(self._add_day_of_month)
            .pipe(self._add_day_of_year)
            .pipe(self._add_week_of_month)
            .pipe(self._add_week_of_year)
            .pipe(self._add_month)
            .pipe(self._add_year)
        )
        return res


class DateIndicator(BaseEstimator, TransformerMixin):
    """
    Enrich a pandas dataframes with a new column indicating if there is a special date.

    This new column will contain a one for each date specified in the `dates` keyword, zero otherwise.

    Parameters
    ----------
    name : str
        The name of the new column. Usually a holiday name such as Easter, Christmas, Black Friday, ...

    dates : List[str]
        A list containing the dates of the holiday. You have to state every holiday explicitly, i.e.
        Christmas from 2018 to 2020 can be encoded as ["2018-12-24", "2019-12-24", "2020-12-24"].

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": range(7)}, index=pd.date_range(start="2019-12-29", periods=7))
    >>> DateIndicator("around_new_year_2020", ["2019-12-31", "2020-01-01", "2020-01-02"]).fit_transform(df)
                A  around_new_year_2020
    2019-12-29  0                     0
    2019-12-30  1                     0
    2019-12-31  2                     1
    2020-01-01  3                     1
    2020-01-02  4                     1
    2020-01-03  5                     0
    2020-01-04  6                     0
    """

    def __init__(self, name: str, dates: List[str]) -> None:
        """Initialize."""
        self.name = name
        self.dates = dates

    def fit(self, X: pd.DataFrame, y=None) -> "DateIndicator":
        """
        Fit the estimator. In this special case, nothing is done.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        DateIndicator
            Fitted transformer.
        """
        self._check_n_features(X, reset=True)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add the new date feature to the dataframe.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas dataframe with a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            The input dataframe with an additional boolean column named `self.name`.
        """
        check_is_fitted(self)
        check_array(X, dtype=None)
        self._check_n_features(X, reset=False)

        return X.assign(**{self.name: lambda df: df.index.isin(self.dates).astype(int)})
