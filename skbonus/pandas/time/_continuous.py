from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
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


class Smoother(BaseContinuousTransformer, ABC):
    """
    Smooth the columns of a data frame by applying a convolution.

    Parameters
    ----------
    frequency : Optional[str]
        A pandas time frequency. Can take values like "d" for day or "m" for month. A full list can
        be found on https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        If None, the transformer tries to infer it during fit time.

    window : int
        Size of the sliding window. The effect of a holiday will reach from approximately
        date - `window/2 * frequency` to date + `window/2 * frequency`, i.e. it is centered around the dates in `dates`.

    mode : str
        Which convolution mode to use. Can be one of

            - "full": The output is the full discrete linear convolution of the inputs.
            - "valid": The output consists only of those elements that do not rely on the zero-padding.
            - "same": The output is the same size as the first input, centered with respect to the 'full' output.
    """

    def __init__(
        self,
        frequency: Optional[str],
        window: int,
        mode: str,
    ) -> None:
        """Initialize."""
        super().__init__(frequency)
        self.window = window
        self.mode = mode

    @abstractmethod
    def _set_sliding_window(self) -> None:
        """
        Calculate the sliding window.

        Returns
        -------
        None
        """

    def fit(self, X: pd.DataFrame, y: None = None) -> Smoother:
        """
        Fit the estimator.

        The frequency is computed and the sliding window is created.

        Parameters
        ----------
        X : pd.DataFrame
            Used for inferring the frequency, if not provided during initialization.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        GeneralGaussianSmoother
            Fitted transformer.
        """
        self._check_n_features(X, reset=True)
        self._set_frequency(X)
        self._set_sliding_window()
        self.sliding_window_ = (
            self.sliding_window_.reshape(-1, 1) / self.sliding_window_.sum()
        )

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
            The input dataframe with an additional column for special dates.
        """
        check_is_fitted(self)
        check_array(X, dtype=None)
        self._check_n_features(X, reset=False)

        extended_index = self._make_continuous_time_index(X)
        convolution = convolve2d(
            X.reindex(extended_index).fillna(0), self.sliding_window_, mode=self.mode
        )

        if self.mode == "full" and self.window > 1:
            convolution = convolution[: -self.window + 1]

        smoothed_dates = pd.DataFrame(
            convolution, index=extended_index, columns=X.columns
        ).reindex(X.index)

        return smoothed_dates


class GeneralGaussianSmoother(Smoother):
    """
    Smooth the columns of a data frame by applying a convolution with a generalized Gaussian curve.

    Parameters
    ----------
    frequency : Optional[str], default=None
        A pandas time frequency. Can take values like "d" for day or "m" for month. A full list can
        be found on https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        If None, the transformer tries to infer it during fit time.

    window : int, default=1
        Size of the sliding window. The effect of a holiday will reach from approximately
        date - `window/2 * frequency` to date + `window/2 * frequency`, i.e. it is centered around the dates in `dates`.

    p : float, default=1
        Parameter for the shape of the curve. p=1 yields a typical Gaussian curve while p=0.5 yields a Laplace curve, for example.

    sig : float, default=1
        Parameter for the standard deviation of the bell-shaped curve.

    tails : str, default="both"
        Which tails to use. Can be one of

            - "left"
            - "right"
            - "both"

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [0, 0, 0, 1, 0, 0, 0]}, index=pd.date_range(start="2019-12-29", periods=7))
    >>> GeneralGaussianSmoother().fit_transform(df)
                  A
    2019-12-29  0.0
    2019-12-30  0.0
    2019-12-31  0.0
    2020-01-01  1.0
    2020-01-02  0.0
    2020-01-03  0.0
    2020-01-04  0.0

    >>> GeneralGaussianSmoother(frequency="d", window=5, p=1, sig=1).fit_transform(df)
                       A
    2019-12-29  0.000000
    2019-12-30  0.054489
    2019-12-31  0.244201
    2020-01-01  0.402620
    2020-01-02  0.244201
    2020-01-03  0.054489
    2020-01-04  0.000000

    >>> GeneralGaussianSmoother(window=7, tails="right").fit_transform(df)
                       A
    2019-12-29  0.000000
    2019-12-30  0.000000
    2019-12-31  0.000000
    2020-01-01  0.570459
    2020-01-02  0.346001
    2020-01-03  0.077203
    2020-01-04  0.006337
    """

    def __init__(
        self,
        frequency: Optional[str] = None,
        window: int = 1,
        p: float = 1,
        sig: float = 1,
        tails: str = "both",
    ) -> None:
        """Initialize."""
        super().__init__(frequency, window, mode="same")
        self.p = p
        self.sig = sig
        self.tails = tails

    def _set_sliding_window(self) -> None:
        """
        Calculate the sliding window.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided value for `tails` is not "left", "right" or "both".
        """
        self.sliding_window_ = np.exp(
            -0.5
            * np.abs(np.arange(-self.window // 2 + 1, self.window // 2 + 1) / self.sig)
            ** (2 * self.p)
        )
        if self.tails == "left":
            self.sliding_window_[self.window // 2 + 1 :] = 0
        elif self.tails == "right":
            self.sliding_window_[: self.window // 2] = 0
        elif self.tails != "both":
            raise ValueError(
                "tails keyword has to be one of 'both', 'left' or 'right'."
            )


class ExponentialDecaySmoother(Smoother):
    """
    Smooth the columns of a data frame by applying a convolution with a exponentially decaying curve.

    This class can be used for modelling carry over effects in marketing mix models

    Parameters
    ----------
    frequency : Optional[str], default=None
        A pandas time frequency. Can take values like "d" for day or "m" for month. A full list can
        be found on https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        If None, the transformer tries to infer it during fit time.

    window : int, default=1
        Size of the sliding window. The effect of a holiday will reach from approximately
        date - `window/2 * frequency` to date + `window/2 * frequency`, i.e. it is centered around the dates in `dates`.

    strength : float, default=0.0
        Fraction of the spending effect that is carried over.

    peak : float, default=0.0
        Where the carryover effect peaks.

    exponent : float, default=1.0
        To further widen or narrow the carryover curve. A value of 1.0 yields a normal exponential decay.
        With values larger than 1.0, a super exponential decay can be achieved.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [0, 0, 0, 1, 0, 0, 0]}, index=pd.date_range(start="2019-12-29", periods=7))
    >>> ExponentialDecaySmoother().fit_transform(df)
                  A
    2019-12-29  0.0
    2019-12-30  0.0
    2019-12-31  0.0
    2020-01-01  1.0
    2020-01-02  0.0
    2020-01-03  0.0
    2020-01-04  0.0

    >>> ExponentialDecaySmoother(frequency="d", window=3, strength=0.5).fit_transform(df)
                       A
    2019-12-29  0.000000
    2019-12-30  0.000000
    2019-12-31  0.000000
    2020-01-01  0.571429
    2020-01-02  0.285714
    2020-01-03  0.142857
    2020-01-04  0.000000

    >>> ExponentialDecaySmoother(window=3, strength=0.5, peak=1).fit_transform(df)
                   A
    2019-12-29  0.00
    2019-12-30  0.00
    2019-12-31  0.00
    2020-01-01  0.25
    2020-01-02  0.50
    2020-01-03  0.25
    2020-01-04  0.00
    """

    def __init__(
        self,
        frequency: Optional[str] = None,
        window: int = 1,
        strength: float = 0.0,
        peak: float = 0.0,
        exponent: float = 1.0,
    ) -> None:
        """Initialize."""
        super().__init__(frequency, window, mode="full")
        self.strength = strength
        self.peak = peak
        self.exponent = exponent

    def _set_sliding_window(self) -> None:
        """
        Calculate the sliding window.

        Returns
        -------
        None
        """
        self.sliding_window_ = self.strength ** (
            np.abs(np.arange(self.window) - self.peak) ** self.exponent
        )
