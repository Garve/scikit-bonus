"""Naive outlier methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted


class BoxEnvelope(BaseEstimator, OutlierMixin, ABC):
    """
    Detect if a data point is an outlier via checking each feature for unusual behavior independently.

    It works the following way for each sample:
        - Mark the sample as an inlier
        - For each feature, do the following:
            - If the feature is smaller than the `alpha` / 2 quantile, mark this sample as an outlier.
            - If the feature is larger than the 1 - `alpha` / 2 quantile, mark this sample as an outlier.

    """

    @abstractmethod
    def _get_bounds(self, X):
        """Implement this. This should set `self.lower_bounds_` and `self.upper_bounds`."""

    def fit(self, X: np.ndarray, y=None) -> BoxEnvelope:
        """
        Fit the estimator.

        Parameters
        ----------
        X : np.ndarray
            Used for calculating the quantiles.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        BoxEnvelope
            Fitted transformer.
        """
        X = check_array(X)
        self._check_n_features(X, reset=True)

        self._get_bounds(X)
        self.offset_ = 0

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels (1 inlier, -1 outlier) of X according to the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The data.

        Returns
        -------
        np.ndarray
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        return (
            2
            * np.prod(
                np.hstack(
                    [
                        np.where(X < self.lower_bounds_, 0, 1),
                        np.where(X > self.upper_bounds_, 0, 1),
                    ]
                ),
                axis=1,
            )
            - 1
        ).astype(float)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels (1 inlier, -1 outlier) of X according to the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The data.

        Returns
        -------
        np.ndarray
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        return self.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels (1 inlier, -1 outlier) of X according to the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The data.

        Returns
        -------
        np.ndarray
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        return self.decision_function(X).astype(int)


class QuantileBoxEnvelope(BoxEnvelope):
    """
    Detect if a data point is an outlier via checking each feature for unusual behavior independently.

    It works the following way for each sample:
        - Mark the sample as an inlier
        - For each feature, do the following:
            - If the feature is smaller than the `alpha` / 2 quantile, mark this sample as an outlier.
            - If the feature is larger than the 1 - `alpha` / 2 quantile, mark this sample as an outlier.

    Parameters
    ----------
    alpha : float, default=0.01
        Determines how many outliers are produced. The larger `alpha`, the more samples are marked as outliers.
        For one-dimensional data, approximately a fraction `alpha` of all samples will be marked as outliers.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([
    ...     [1, 5],
    ...     [2, 5],
    ...     [3, 5],
    ...     [2, 6]
    ... ])
    >>> QuantileBoxEnvelope(alpha=0.4).fit_predict(X)
    array([-1,  1, -1, -1])

    >>> np.random.seed(0)
    >>> X = np.random.randn(1000, 1)
    >>> list(QuantileBoxEnvelope().fit_predict(X)).count(-1) / 1000
    0.01

    """

    def __init__(self, alpha: float = 0.01):
        """Initialize."""
        super().__init__()
        self.alpha = alpha

    def _get_bounds(self, X):
        self.lower_bounds_ = np.quantile(X, q=self.alpha / 2, axis=0)
        self.upper_bounds_ = np.quantile(X, q=1 - self.alpha / 2, axis=0)
