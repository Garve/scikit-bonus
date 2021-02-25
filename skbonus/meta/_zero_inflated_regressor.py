from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from skbonus.utils.validation import check_n_features


class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):
    """
    A meta regressor for zero-inflated datasets, i.e. the targets contain a lot of zeroes.

    `ZeroInflatedRegressor` consists of a classifier and a regressor.

        - The classifier's task is to find of if the target is zero or not.
        - The regressor's task is to output a (usually positive) prediction whenever the classifier indicates that the there should be a non-zero prediction.

    The regressor is only trained on examples where the target is non-zero, which makes it easier for it to focus.

    At prediction time, the classifier is first asked if the output should be zero. If yes, output zero.
    Else, ask the regressor for its prediction and output it.

    Parameters
    ----------
    classifier : Any, scikit-learn classifier, default=`LogisticRegression()`
        A classifier that answers the question "Should the output be zero?".

    regressor : Any, scikit-learn regressor, default=`LinearRegression()`
        A regressor for predicting the target. Its prediction is only used if `classifier` says that the output is non-zero.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import ExtraTreesClassifier
    >>> np.random.seed(0)
    >>> X = np.random.randn(10000, 4)
    >>> y = ((X[:, 0]>0) & (X[:, 1]>0)) * np.abs(X[:, 2] * X[:, 3]**2)
    >>> z = ZeroInflatedRegressor(classifier=ExtraTreesClassifier(random_state=0))
    >>> z.fit(X, y)
    ZeroInflatedRegressor(classifier=ExtraTreesClassifier(random_state=0))
    >>> z.predict(X)[:5]
    array([0.52977797, 0.        , 0.        , 0.73678052, 0.        ])


    """

    def __init__(self, classifier: Any = None, regressor: Any = None) -> None:
        """Initialize."""
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X: np.array, y: np.array) -> "ZeroInflatedRegressor":
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
        self.n_features_in_ = X.shape[1]

        self.classifier_ = (
            clone(self.classifier)
            if self.classifier is not None
            else LogisticRegression()
        )
        self.classifier_.fit(X, y > 0)

        positive_indices = self.classifier_.predict(X) == 1
        X_pos = X[positive_indices]
        y_pos = y[positive_indices]

        self.regressor_ = (
            clone(self.regressor) if self.regressor is not None else LinearRegression()
        )
        self.regressor_.fit(X_pos, y_pos)

        return self

    def predict(self, X: np.array) -> np.array:
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
        check_n_features(self, X)

        return self.classifier_.predict(X) * self.regressor_.predict(X)
