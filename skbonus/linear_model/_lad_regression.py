import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    _check_sample_weight,
)


class LADRegression(BaseEstimator, RegressorMixin):
    """
    Least absolute deviation Regression.

    LADRegression fits a linear model to minimize the residual sum of absolute deviations between
    the observed targets in the dataset, and the targets predicted by the linear approximation.

    Compared to linear regression, this approach is robust to outliers. You can even
    optimize for the lowest MAPE (Mean Average Percentage Error), if you pass in np.abs(1/y_train) for the
    sample_weight keyword when fitting the regressor.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    coef_ : np.array of shape (n_features,)
        Estimated coefficients of the model.

    intercept_ : float
        Independent term in the linear model. Set to 0.0 if fit_intercept = False.

    Notes
    -----
    This implementation uses scipy.optimize.minimize, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 4)
    >>> y = X @ np.array([1, 2, 3, 4])
    >>> l = LADRegression()
    >>> l.fit(X, y)
    LADRegression()
    >>> l.coef_
    array([1.        , 1.99999999, 3.        , 3.99999999])

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 4)
    >>> y = X @ np.array([-1, 2, -3, 4])
    >>> l = LADRegression(positive=True)
    >>> l.fit(X, y)
    LADRegression(positive=True)
    >>> l.coef_
    array([0.        , 1.42653583, 0.        , 4.29939267])

    """

    def __init__(
        self, fit_intercept: bool = True, copy_X: bool = True, positive: bool = False
    ) -> None:
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.positive = positive

    def fit(
        self,
        X: np.array,
        y: np.array,
        coef_init: np.array = None,
        intercept_init: np.array = None,
        sample_weight: np.array = None,
    ) -> "LADRegression":
        """
        Fit the model.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The training data.

        y : np.array, 1-dimensional
            The target values.

        coef_init : np.array, default=None
            Used for the numerical optimization of the mean absolute error loss function. Initialization value
            for the coefficients.

        intercept_init : np.array, default=None
            Used for the numerical optimization of the mean absolute error loss function. Initialization value
            for the intercept.

        sample_weight : np.array, default=None
            Individual weights for each sample. Use np.abs(1/y_train) to optimize for the lowest
            MAPE (Mean Average Percentage Error).

        Returns
        -------
        self
            Fitted regressor.
        """
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)

        def mae_loss(params):
            return np.mean(sample_weight * np.abs(y - X_ @ params))

        n, d = X.shape

        if self.copy_X:
            X_ = X.copy()
        else:
            X_ = X

        coefs = np.zeros(d) if coef_init is None else coef_init

        if self.fit_intercept:
            X_ = np.hstack([X_, np.ones(shape=(n, 1))])
            intercept = np.zeros(1) if intercept_init is None else intercept_init
            start_params = np.hstack([coefs, intercept])
            bounds = [(0, np.inf) for _ in range(d + 1)] if self.positive else None

            *self.coef_, self.intercept_ = minimize(
                mae_loss, x0=start_params, bounds=bounds, method="L-BFGS-B"
            ).x
        else:
            start_params = coefs
            bounds = [(0, np.inf) for _ in range(d)] if self.positive else None

            self.coef_ = minimize(
                mae_loss, x0=start_params, bounds=bounds, method="L-BFGS-B"
            ).x
            self.intercept_ = 0.0

        self.coef_ = np.array(self.coef_)

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predict using the linear model.

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

        return X @ self.coef_ + self.intercept_


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
