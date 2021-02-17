from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
import numpy as np


class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):
    """
    A meta regressor for zero-inflated datasets, i.e. the targets contain a lot of zeroes.

    ZeroInflatedRegressor consists of a classifier and a regressor.

        - The classifier's task is to find of if the target is zero or not.
        - The regressor's task is to output a (usually positive) prediction whenever the classifier indicates that the there should be a non-zero prediction.

    The regressor is only trained on examples where the target is non-zero, which makes it easier for it to focus.

    At prediction time, the classifier is first asked if the output should be zero. If yes, output zero.
    Else, ask the regressor for its prediction and output it.

    Parameters
    ----------
    classifier : Any, scikit-learn classifier
        A classifier that answers the question "Should the output be zero?". A threshold can be passed with the `threshold` keyword.

    regressor : Any, scikit-learn regressor
        A regressor for predicting the target. Its prediction is only used if `classifier` says that the output is non-zero.

    threshold : float, default=0.5
        Threshold for the classifier.
    """

    def __init__(self, classifier: Any, regressor: Any, threshold: float = 0.5) -> None:
        """Initialize."""
        self.classifier = classifier
        self.regressor = regressor
        self.threshold = threshold

    def fit(self, X: np.array, y: np.array):
        """
        Fit the model.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The training data.

        y : np.array, 1-dimensional
            The target values.

        Returns
        -------
        ZeroInflatedRegressor
            Fitted regressor.
        """
        X, y = check_X_y(X, y)

        self.classifier.fit(X, y > 0)
        positive_indices = self.classifier.predict_proba(X)[:, 1] > self.threshold

        self.regressor.fit(X[positive_indices], y[positive_indices])

    def predict(self, X: np.array):
        """
        Get predictions.

        Parameters
        ----------
        X : np.array, shape (n_samples, n_features)
            Samples to get predictions of.

        Returns
        -------
        y : np.array, shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)

        return (self.classifier.predict_proba(X)[:, 1] > self.threshold) * self.regressor.predict(X)
