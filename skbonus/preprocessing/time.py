from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
from scipy.signal import convolve2d
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Break each cyclic feature into two new features, corresponding to the representation of this feature on a circle.

    For example, take the hours from 0 to 23. On a normal, round  analog clock,
    these features are perfectly aligned on a circle already. You can do the same with days, month, ...

    Notes
    -----
    This method has the advantage that close points in time stay close together. See the examples below.

    Otherwise, if algorithms deal with the raw value for hour they cannot know that 0 and 23 are actually close.
    Another possibility is one hot encoding the hour. This has the disadvantage that it breaks the distances
    between different hours. Hour 5 and 16 have the same distance as hour 0 and 23 when doing this.

    Parameters
    ----------
    cycles : Optional[List[Tuple[float, float]]], default=None
        Define the ranges of the cycles in the format [(col_1_min, col_1_max), (col_2_min, col_2_max), ...).
        For example, use [(0, 23), (1, 7)] if your dataset consists of two columns, the first one containing hours and the second one the day of the week.

        If left empty, the encoder tries to infer it from the data, i.e. it looks for the minimum and maximum value of each column.

    Examples
    --------
    >>> import numpy as np
    >>> df = np.array([[22], [23], [0], [1], [2]])
    >>> CyclicalEncoder().fit_transform(df)
    array([[ 0.8660254 , -0.5       ],
           [ 0.96592583, -0.25881905],
           [ 1.        ,  0.        ],
           [ 0.96592583,  0.25881905],
           [ 0.8660254 ,  0.5       ]])
    """

    def __init__(
        self,
        cycles: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initialize."""
        self.cycles = cycles

    def fit(self, X: np.ndarray, y=None) -> CyclicalEncoder:
        """
        Fit the estimator. In this special case, nothing is done.

        Parameters
        ----------
        X : np.ndarray
            Used for inferring the ranges of the data, if not provided during initialization.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        CyclicalEncoder
            Fitted transformer.
        """
        X = check_array(X)
        self._check_n_features(X, reset=True)

        if self.cycles is None:
            self.cycles_ = list(zip(X.min(axis=0), X.max(axis=0)))
        else:
            self.cycles_ = self.cycles

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Add the cyclic features to the dataframe.

        Parameters
        ----------
        X : np.ndarray
            The data with cyclical features in the columns.

        Returns
        -------
        np.ndarray
            The encoded data with twice as man columns as the original.
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        def min_max(column):
            return (
                (column - self.cycles_[i][0])
                / (self.cycles_[i][1] + 1 - self.cycles_[i][0])
                * 2
                * np.pi
            )

        res = []

        for i in range(X.shape[1]):
            res.append(np.cos(min_max(X[:, i])))
            res.append(np.sin(min_max(X[:, i])))

        return np.vstack(res).T


class Smoother(BaseEstimator, TransformerMixin, ABC):
    """
    Smooth the columns of an array by applying a convolution.

    Parameters
    ----------
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
        window: int,
        mode: str,
    ) -> None:
        """Initialize."""
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

    def fit(self, X: np.ndarray, y: None = None) -> Smoother:
        """
        Fit the estimator.

        The frequency is computed and the sliding window is created.

        Parameters
        ----------
        X : np.ndarray
            Used for inferring the frequency, if not provided during initialization.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        GeneralGaussianSmoother
            Fitted transformer.
        """
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self._set_sliding_window()
        self.sliding_window_ = (
            self.sliding_window_.reshape(-1, 1) / self.sliding_window_.sum()
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Add the new date feature to the dataframe.

        Parameters
        ----------
        X : np.ndarray
            A pandas dataframe with a DatetimeIndex.

        Returns
        -------
        np.ndarray
            The input dataframe with an additional column for special dates.
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        convolution = convolve2d(X, self.sliding_window_, mode=self.mode)

        if self.mode == "full" and self.window > 1:
            convolution = convolution[: -self.window + 1]

        return convolution


class GeneralGaussianSmoother(Smoother):
    """
    Smooth the columns of an array by applying a convolution with a generalized Gaussian curve.

    Parameters
    ----------
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
    >>> import numpy as np
    >>> X = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
    >>> GeneralGaussianSmoother().fit_transform(X)
    array([[0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.]])

    >>> GeneralGaussianSmoother(window=5, p=1, sig=1).fit_transform(X)
    array([[0.        ],
           [0.05448868],
           [0.24420134],
           [0.40261995],
           [0.24420134],
           [0.05448868],
           [0.        ]])

    >>> GeneralGaussianSmoother(window=7, tails="right").fit_transform(X)
    array([[0.        ],
           [0.        ],
           [0.        ],
           [0.57045881],
           [0.34600076],
           [0.0772032 ],
           [0.00633722]])
    """

    def __init__(
        self,
        window: int = 1,
        p: float = 1,
        sig: float = 1,
        tails: str = "both",
    ) -> None:
        """Initialize."""
        super().__init__(window, mode="same")
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
    Smooth the columns of an array by applying a convolution with a exponentially decaying curve.

    This class can be used for modelling carry over effects in marketing mix models.

    Parameters
    ----------
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
    >>> import numpy as np
    >>> X = np.array([0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
    >>> ExponentialDecaySmoother().fit_transform(X)
    array([[0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.]])

    >>> ExponentialDecaySmoother(window=3, strength=0.5).fit_transform(X)
    array([[0.        ],
           [0.        ],
           [0.        ],
           [0.57142857],
           [0.28571429],
           [0.14285714],
           [0.        ]])

    >>> ExponentialDecaySmoother(window=3, strength=0.5, peak=1).fit_transform(X)
    array([[0.  ],
           [0.  ],
           [0.  ],
           [0.25],
           [0.5 ],
           [0.25],
           [0.  ]])
    """

    def __init__(
        self,
        window: int = 1,
        strength: float = 0.0,
        peak: float = 0.0,
        exponent: float = 1.0,
    ) -> None:
        """Initialize."""
        super().__init__(window, mode="full")
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
