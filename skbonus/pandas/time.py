from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    This class enriches pandas dataframes with a DatetimeIndex with new columns. These new columns are easy
    derivations from the index, such as the day of week or month.
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
        Fit the estimator. In this special case, nothing is done.

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
    Adds a power trend to a pandas dataframe with a continous DatetimeIndex. For example, it can create a new column
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
    ...     index=pd.date_range(start='1988-08-08', periods=4)
    ... )
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
        Fits the model. It assigns value 0 to the first item of the time index and 1 to the second one etc.
        This way, we can get a value for any other date in a linear fashion. These values are later transformed.

        Raises a ValueError if the DatetimeIndex has no frequency.

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
        freq = X.index.freq
        if freq is None:
            raise ValueError(
                "DatetimeIndex is not continuous. Frequency could not be calculated."
            )
        t1 = X.index.min().value
        t2 = (X.index.min() + freq).value

        self.date_to_index_ = lambda x: (x - t1) / (t2 - t1)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add the trend column to the input dataframe.

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

        return X.assign(trend=self.date_to_index_(index) ** self.power)


class SpecialDatesAdder(BaseEstimator, TransformerMixin):
    """
    This class enriches pandas dataframes with a DatetimeIndex with new columns. These new columns
    contain whether the index lies within a time interval. For example, the output can be
    a one hot encoded column containing a 1 if the corresponding date from the index is within
    the given date range, and 0 otherwise.
    The output can also be a smoothed by sliding a window over the one hot encoded column as a next step.
    This makes sense when, for example, a certain holiday has effects on the next days or the days before, too.
    See the examples to get a better understanding.

    This is especially useful when dealing with time series regressions or classifications.

    Parameters
    ----------
    name : str
        The name of the new column. Usually a holiday name such as Easter, Christmas, Black Friday, ...

    dates : List[Union[pd.Timestamp, str]]
        A list containing the dates of the holiday. You have to state every holiday explicitly, i.e.
        Christmas from 2018 to 2020 can be encoded as ['2018-12-24', '2019-12-24', '2020-12-24'].

    window : int, default=1
        Size of the sliding window. Used for smoothing the simple one hot encoded output. Increasing
        it to something larger than 1 only makes sense for a DatetimeIndex with equidistant dates.

    center : bool, default=False
        Whether the window is centered. If True, a window of size 5 at time t includes the times
        t-2, t-1, t, t+1 and t+2. Useful if the effect of a holiday can be seen before the holiday already.
        If False, the window would include t-4, t-3, t-2, t-1, t. Useful if the effect of a holiday starts
        exactly with the holiday and wears off over time.

    win_type : Optional[str], default=None
        Type of smoothing. A value of None leaves the default one hot encoding, i.e. the output column
        contains 0 and 1 only. Another interesting window is 'gaussian', which also requires the parameter std.
        See the notes below for further information.

    pad_value : Union[float, np.nan], default=0.
        When using sliding windows of length > 1, the time series has to be extended to prevent NaNs at
        the start or end of the smoothed time series. If you wish for these NaNs, pad_value=input np.nan.
        See the examples below for further information.

    window_function_kwargs : Optional[Any]
        Settings for certain win_type functions, for example std if win_type='gaussian'.

    Notes
    -----
    win_type accepts a string of any scipy.signal window function, see
    https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': range(7)}, index=pd.date_range(start='2019-12-29', periods=7))
    >>> sda = SpecialDatesAdder('new_year_2020', ['2020-01-01'])
    >>> sda.fit_transform(df)
                A   new_year_2020
    2019-12-29  0             0.0
    2019-12-30  1             0.0
    2019-12-31  2             0.0
    2020-01-01  3             1.0
    2020-01-02  4             0.0
    2020-01-03  5             0.0
    2020-01-04  6             0.0

    >>> smooth_sda = SpecialDatesAdder('new_year_2020', ['2020-01-01'],
    ... window=5, center=True, win_type='gaussian', std=1)
    >>> smooth_sda.fit_transform(df)
                A   new_year_2020
    2019-12-29  0        0.000000
    2019-12-30  1        0.135335
    2019-12-31  2        0.606531
    2020-01-01  3        1.000000
    2020-01-02  4        0.606531
    2020-01-03  5        0.135335
    2020-01-04  6        0.000000

    >>> smooth_sda = SpecialDatesAdder('new_year_2020', ['2020-01-01'],
    ... window=5, center=True, win_type='gaussian', std=1, pad_value=np.nan)
    >>> smooth_sda.fit_transform(df)
                A   new_year_2020
    2019-12-29  0             NaN
    2019-12-30  1             NaN
    2019-12-31  2        0.606531
    2020-01-01  3        1.000000
    2020-01-02  4        0.606531
    2020-01-03  5             NaN
    2020-01-04  6             NaN
    """

    def __init__(
        self,
        name: str,
        dates: List[Union[pd.Timestamp, str]],
        window: int = 1,
        center: bool = False,
        win_type: Optional[str] = None,
        pad_value: Union[float, "np.nan"] = 0,
        **window_function_kwargs
    ) -> None:
        self.name = name
        self.dates = dates
        self.window = window
        self.center = center
        self.win_type = win_type
        self.pad_value = pad_value
        self.window_function_kwargs = window_function_kwargs

    def fit(self, X: pd.DataFrame, y: None = None) -> "SpecialDatesAdder":
        """
        Fit the estimator. In this special case, nothing is done.

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

        Parameters
        ----------
        X : pd.DataFrame
            A pandas dataframe with a DatetimeIndex.

        Returns
        -------
        transformed_X : pd.DataFrame
            A pandas dataframe with an additional column for special dates.
        """
        dummy_dates = pd.Series(X.index.isin(self.dates), index=X.index)
        extended_index = extended_index = X.index.union(
            [X.index.min() - i * X.index.freq for i in range(1, self.window + 1)]
        ).union([X.index.max() + i * X.index.freq for i in range(1, self.window + 1)])

        smoothed_dates = (
            dummy_dates.reindex(extended_index)
            .fillna(self.pad_value)
            .rolling(window=self.window, center=self.center, win_type=self.win_type)
            .sum(**self.window_function_kwargs)
            .reindex(X.index)
            .values
        )

        return X.assign(**{self.name: smoothed_dates})


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
