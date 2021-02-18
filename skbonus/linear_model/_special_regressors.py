from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
)


class BaseScipyMinimizeRegressor(BaseEstimator, RegressorMixin, ABC):
    """
    Base class for regressors relying on scipy's minimze method. Derive a class from this one and give it the function to be minimized.

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
    """

    def __init__(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        positive: bool = False,
    ) -> None:
        """Initialize."""
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.positive = positive

    @abstractmethod
    def _get_objective(
        self, X: np.array, y: np.array, sample_weight: np.array
    ) -> Tuple[Callable[[np.array], float], Callable[[np.array], np.array]]:
        """
        Produce the loss function to be minimized, and its gradient to speed up computations.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The training data.

        y : np.array, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        loss : Callable[[np.array], float]
            The loss function to be minimized.

        grad_loss : Callable[[np.array], np.array]
            The gradient of the loss function. Speeds up finding the minimum.
        """
        pass

    def _loss_regularize(self, loss):
        def regularized_loss(params):
            return (
                loss(params)
                + self.alpha * self.l1_ratio * np.sum(np.abs(params))
                + 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(params ** 2)
            )

        return regularized_loss

    def _grad_loss_regularize(self, grad_loss):
        def regularized_grad_loss(params):
            return (
                grad_loss(params)
                + self.alpha * self.l1_ratio * np.sign(params)
                + self.alpha * (1 - self.l1_ratio) * params
            )

        return regularized_grad_loss

    def fit(
        self,
        X: np.array,
        y: np.array,
        sample_weight: Optional[np.array] = None,
    ) -> "BaseScipyMinimizeRegressor":
        """
        Fit the model using the SLSQP algorithm.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The training data.

        y : np.array, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        Fitted regressor.
        """
        X_, grad_loss, loss = self._prepare_inputs(X, sample_weight, y)

        d = X_.shape[1]
        bounds = [(0, np.inf) for _ in range(d)] if self.positive else None
        minimize_result = minimize(
            loss,
            x0=np.zeros(d),
            bounds=bounds,
            method="SLSQP",
            jac=grad_loss,
            tol=1e-20,
        )
        self.convergence_status_ = minimize_result.message

        if self.fit_intercept:
            *self.coef_, self.intercept_ = minimize_result.x
        else:
            self.coef_ = minimize_result.x
            self.intercept_ = 0.0

        self.coef_ = np.array(self.coef_)

        return self

    def _prepare_inputs(self, X, sample_weight, y):
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)
        self.n_features_in_ = X.shape[1]

        n = X.shape[0]
        if self.copy_X:
            X_ = X.copy()
        else:
            X_ = X
        if self.fit_intercept:
            X_ = np.hstack([X_, np.ones(shape=(n, 1))])

        loss, grad_loss = self._get_objective(X_, y, sample_weight)

        return X_, grad_loss, loss

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


class LADRegression(BaseScipyMinimizeRegressor):
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
    >>> l = LADRegression().fit(X, y)
    >>> l.coef_
    array([1., 2., 3., 4.])

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 4)
    >>> y = X @ np.array([-1, 2, -3, 4])
    >>> l = LADRegression(positive=True).fit(X, y)
    >>> l.coef_
    array([7.39575926e-18, 1.42423304e+00, 2.80467827e-17, 4.29789588e+00])

    """

    def _get_objective(
        self, X: np.array, y: np.array, sample_weight: np.array
    ) -> Tuple[Callable[[np.array], float], Callable[[np.array], np.array]]:
        @self._loss_regularize
        def mae_loss(params):
            return np.mean(sample_weight * np.abs(y - X @ params))

        @self._grad_loss_regularize
        def grad_mae_loss(params):
            return -(sample_weight * np.sign(y - X @ params)) @ X / X.shape[0]

        return mae_loss, grad_mae_loss


class ImbalancedLinearRegression(BaseScipyMinimizeRegressor):
    """
    Linear regression where overestimating is `overestimation_punishment_factor` times worse than underestimating.

    A value of `overestimation_punishment_factor=5` implies that overestimations by the model are penalized with a factor of 5
    while underestimations have a default factor of 1.

    ImbalancedLinearRegression fits a linear model to minimize the residual sum of squares between
    the observed targets in the dataset, and the targets predicted by the linear approximation.
    Compared to normal linear regression, this approach allows for a different treatment of over or under estimations.

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

    overestimation_punishment_factor : float, default=1
        Factor to punish overestimations more (if the value is larger than 1) or less (if the value is between 0 and 1).

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
    >>> y = X @ np.array([1, 2, 3, 4]) + 2*np.random.randn(100)
    >>> over_bad = ImbalancedLinearRegression(overestimation_punishment_factor=50).fit(X, y)
    >>> over_bad.coef_
    array([0.36267036, 1.39526844, 3.4247146 , 3.93679175])

    >>> under_bad = ImbalancedLinearRegression(overestimation_punishment_factor=0.01).fit(X, y)
    >>> under_bad.coef_
    array([0.73519586, 1.28698197, 2.61362614, 4.35989806])

    """

    def __init__(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        positive: bool = False,
        overestimation_punishment_factor: float = 1.0,
    ) -> None:
        """Initialize."""
        super().__init__(alpha, l1_ratio, fit_intercept, copy_X, positive)
        self.overestimation_punishment_factor = overestimation_punishment_factor

    def _get_objective(
        self, X: np.array, y: np.array, sample_weight: np.array
    ) -> Tuple[Callable[[np.array], float], Callable[[np.array], np.array]]:
        @self._loss_regularize
        def imbalanced_loss(params):
            return 0.5 * np.mean(
                sample_weight
                * np.where(X @ params > y, self.overestimation_punishment_factor, 1)
                * np.square(y - X @ params)
            )

        @self._grad_loss_regularize
        def grad_imbalanced_loss(params):
            return (
                -(
                    sample_weight
                    * np.where(X @ params > y, self.overestimation_punishment_factor, 1)
                    * (y - X @ params)
                )
                @ X
                / X.shape[0]
            )

        return imbalanced_loss, grad_imbalanced_loss
