from __future__ import annotations
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class BaseContinuousTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for dealing with continuous Datetime indices.

    Parameters
    ----------
    frequency : Optional[str]
        A pandas time frequency. Can take values like "d" for day or "m" for month. A full list can
        be found on https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        If None, the transformer tries to infer it during fit time.
    """

    def __init__(self, frequency) -> None:
        """Initialize."""
        self.frequency = frequency

    def _set_frequency(self, X: pd.DataFrame) -> None:
        """
        Infer the frequency.

        Parameters
        ----------
        X : pd.DataFrame
            Input fataframe

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no frequency was provided during initialization and also cannot be inferred.

        """
        if self.frequency is None:
            self.freq_ = X.index.freq
            if self.freq_ is None:
                raise ValueError(
                    "No frequency provided. It was also impossible to infer it while fitting. Please provide a value in the frequency keyword."
                )
        else:
            self.freq_ = pd.tseries.frequencies.to_offset(self.frequency)

    def _make_continuous_time_index(
        self,
        X: pd.DataFrame,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DatetimeIndex:
        """
        Sometimes, the input data has missing time periods. As this is bad for time difference based calculations, create a continuous DatetimeIndex first.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with a DatetimeIndex, potentially with gaps.

        Returns
        -------
        pd.DatetimeIndex
            A continuous time range.
        """
        extended_index = pd.date_range(
            start=X.index.min() if start is None else start,
            end=X.index.max() if end is None else end,
            freq=self.freq_,
        )
        return extended_index


class PowerTrend(BaseContinuousTransformer):
    """
    Add a power trend column to a pandas dataframe.

    For example, it can create a new column with numbers increasing quadratically in the index.

    Parameters
    ----------
    power : float, default=1.0
        Exponent to use for the trend, i.e. linear (power=1.), root (power=0.5), or cube (power=3.).

    frequency : Optional[str], default=None
        A pandas time frequency. Can take values like "d" for day or "m" for month. A full list can
        be found on https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        If None, the transformer tries to infer it during fit time.

    origin_date : Optional[Union[str, pd.Timestamp]], default=None
        A date the trend originates from, i.e. the value of the trend column is zero for this date.
        If None, the transformer uses the smallest date of the training set during fit time.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"A": ["a", "b", "c", "d"]},
    ...     index=pd.date_range(start="1988-08-08", periods=4)
    ... )
    >>> PowerTrend(power=2., frequency="d", origin_date="1988-08-06").fit_transform(df)
                A  trend
    1988-08-08  a    4.0
    1988-08-09  b    9.0
    1988-08-10  c   16.0
    1988-08-11  d   25.0
    """

    def __init__(
        self,
        power: float = 1.0,
        frequency: Optional[str] = None,
        origin_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> None:
        """Initialize."""
        super().__init__(frequency)
        self.power = power
        self.origin_date = origin_date

    def fit(self, X: pd.DataFrame, y: None = None) -> PowerTrend:
        """
        Fit the model.

        The point of origin and the frequency is constructed here, if not provided during initialization.

        Parameters
        ----------
        X : pd.DataFrame
            Used for inferring the frequency and the origin date, if not provided during initialization.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        PowerTrend
            Fitted transformer.
        """
        self._set_frequency(X)
        self._check_n_features(X, reset=True)

        if self.origin_date is None:
            self.origin_ = X.index.min()
        else:
            self.origin_ = pd.Timestamp(self.origin_date, freq=self.freq_)

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
        pd.DataFrame
            The input dataframe with an additional trend column.

        """
        check_is_fitted(self)
        check_array(X, dtype=None)
        self._check_n_features(X, reset=False)

        extended_index = self._make_continuous_time_index(X, start=self.origin_)
        dummy_dates = pd.Series(np.arange(len(extended_index)), index=extended_index)

        return X.assign(trend=dummy_dates.reindex(X.index) ** self.power)
