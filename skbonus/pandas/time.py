import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any


class DateFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    This class enriches pandas dataframes with a DatetimeIndex with new columns.
    This is especially useful when dealing with time series regressions or classifications.

    Parameters
    ----------
    day_of_week : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    day_of_month : bool, default=True
        Whether to extract the day of month from the index and add it as a new column.

    day_of_year : bool, default=False
        Whether to extract the day of year from the index and add it as a new column.

    week_of_month : bool, default=False
        Whether to extract the week of month from the index and add it as a new column.

    week_of_year : bool, default=False
        Whether to extract the week of year from the index and add it as a new column.

    month : bool, default=True
        Whether to extract the month from the index and add it as a new column.

    year : bool, default=True
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
    >>> dfa = DateFeaturesAdder()
    >>> dfa.fit_transform(df)
                A   day_of_month    month    year
    1988-08-08  a              8        8    1988
    2000-01-01  b              1        1    2000
    1950-12-31  c             31       12    1950
    """

    def __init__(
        self,
        day_of_week: bool = False,
        day_of_month: bool = True,
        day_of_year: bool = False,
        week_of_month: bool = False,
        week_of_year: bool = False,
        month: bool = True,
        year: bool = True,
    ) -> None:
        self.day_of_week = day_of_week
        self.day_of_month = day_of_month
        self.day_of_year = day_of_year
        self.week_of_month = week_of_month
        self.week_of_year = week_of_year
        self.month = month
        self.year = year

    def _add_day_of_week(self, X: pd.DataFrame) -> pd.DataFrame:
        return (
            X.assign(day_of_week=lambda df: df.index.weekday) if self.day_of_week else X
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
            X.assign(week_of_year=lambda df: df.index.weekofyear)
            if self.week_of_year
            else X
        )

    def _add_month(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(month=lambda df: df.index.month) if self.month else X

    def _add_year(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(year=lambda df: df.index.year) if self.year else X

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DateFeaturesAdder":
        """

        Parameters
        ----------
        X : pd.DataFrame
            A pandas dataframe with a DatetimeIndex.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inserts all chosen time features as new columns into the dataframe and outputs it.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas dataframe with a DatetimeIndex.

        Returns
        -------
        transformed_X : pd.DataFrame
            A pandas dataframe with additional time feature columns.

        """
        res = (
            X.pipe(self._add_day_of_week)
            .pipe(self._add_day_of_month)
            .pipe(self._add_day_of_year)
            .pipe(self._add_week_of_month)
            .pipe(self._add_week_of_year)
            .pipe(self._add_month)
            .pipe(self._add_year)
        )
        return res


class PowerTrendAdder(BaseEstimator, TransformerMixin):
    """
    Adds a power trend to a pandas dataframe with a DatetimeIndex. For example, it can create a new column
    with numbers increasing quadratically in the index.

    Parameters
    ----------
    power : float
        Exponent to use for the trend, i.e. linear (power=1), root (power=0.5), or cube (power=3).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"A": ["a", "b", "c", "d"]},
    ...     index=[
    ...         pd.Timestamp("1988-08-08"),
    ...         pd.Timestamp("1988-08-09"),
    ...         pd.Timestamp("1988-08-10"),
    ...         pd.Timestamp("1988-08-11"),
    ...     ])
    >>> pta = PowerTrendAdder(power=2)
    >>> pta.fit_transform(df)
                A   trend
    1988-08-08  a     0.0
    1988-08-09  b     1.0
    1988-08-10  c     4.0
    1988-08-11  d     9.0
    """

    def __init__(self, power: float = 1) -> None:
        self.power = power

    def fit(self, X: pd.DataFrame, y: None = None) -> "PowerTrendAdder":
        """
        Fits the model. It assigns value 0 to the first item of the time index and 1 to the second one.
        This way, we can get a value for any other date in a linear fashion.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas dataframe with a DatetimeIndex.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted transformer.
        """
        smallest_dates = X.index[np.argsort(X.index)[:2]]
        t1 = smallest_dates[0].value
        t2 = smallest_dates[1].value

        self.mapper_ = lambda x: (x - t1) / (t2 - t1)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add the trend column to the input dataframe

        Parameters
        ----------
        X : pd.DataFrame
             A pandas dataframe with a DatetimeIndex.

        Returns
        -------
        transformed_X : pd.DataFrame
            The dataframe with an additional trend column.

        """
        index = X.index.astype(int)

        return X.assign(trend=self.mapper_(index) ** self.power)


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
