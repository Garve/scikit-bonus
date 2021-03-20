"""Remove outliers from time series data."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EllipticEnvelope
from sklearn.utils.validation import check_is_fitted


class SpikeRemover(BaseEstimator, TransformerMixin):
    """
    This class takes a time series and removes outlier spikes.

    It does so by filtering out observations that are not close to their neighbors and replaces them with the
    mean of the neighbors. The amount of spikes being flattened is determined by the `contamination` parameter.

    Parameters
    ----------
    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
        Range is (0, 0.5).

    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling the data. Pass an int for reproducible results
        across multiple function calls.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> y = np.sin(np.linspace(0, 2*np.pi, 100)) + 0.1*np.random.randn(100); y[4] = 5; y[60] = -10
    >>> y_new = SpikeRemover().fit_transform(y.reshape(-1, 1))
    >>> y_new[[3, 4, 5]]
    array([[0.41334056],
           [0.31382311],
           [0.21430566]])
    >>> y_new[[59, 60, 61]]
    array([[-0.60333398],
           [-0.65302915],
           [-0.70272432]])
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        """Initialize."""
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, y: np.ndarray) -> SpikeRemover:
        """
        Fit the estimator.

        Parameters
        ----------
        y : np.ndarray
            A time series containing outlier spikes.

        Returns
        -------
        SpikeRemover
            Fitted transformer.
        """
        self.outlier_detector_ = EllipticEnvelope(
            contamination=self.contamination, random_state=self.random_state
        )
        y_diff = y[1:] - y[:-1]
        self.outlier_detector_.fit(y_diff)

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Remove outliers from the time series.

        Parameters
        ----------
        y : np.ndarray
            The original time series.

        Returns
        -------
        np.ndarray
            Time series without outlier spikes.
        """
        check_is_fitted(self)
        y_copy = y.copy()

        outlier_markers = np.convolve(
            self.outlier_detector_.predict(y_copy[1:] - y_copy[:-1]), np.array([1, 1])
        )
        central_spikes = np.where(outlier_markers == -2)[0]
        border_spikes = np.where(outlier_markers == -1)[0]

        for spike in central_spikes:
            y_copy[spike] = np.average(y_copy[[spike - 1, spike + 1]])

        for spike in border_spikes:
            if spike == 0:
                y_copy[0] = y_copy[1]
            else:
                y_copy[-1] = y_copy[-2]

        return y_copy
