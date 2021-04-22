from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import (
    check_X_y,
    check_is_fitted,
    check_array,
    _check_sample_weight,
)

from . import utils
from interpret.glassbox.ebm.utils import EBMUtils
from interpret.glassbox.ebm.ebm import EBMExplanation


class ExplainableBoostingMetaRegressor(BaseEstimator, RegressorMixin):
    """
    A meta regressor that outputs a transparent, explainable model given blackbox models.

    It works exactly like the `ExplainableBoostingRegressor` by the interpretml team, but here you can choose any base regressor instead of
    being restricted to trees. For example, you can use scikit-learn's `IsotonicRegression` to create a model that is
    monotonically increasing or decreasing in some of the features, while still being explainable and well-performing.

    See the notes below to find a nice explanation of how the algorithm works at a high level.

    Parameters
    ----------
    base_regressor : Any, default=DecisionTreeRegressor(max_depth=4)
        A single scikit-learn compatible regressor or a list of those regressors of length `n_features`.

    max_rounds : int, default=5000
        Conduct the boosting for these many rounds.

    learning_rate : float, default=0.01
        The learning rate. Should be quite small.

    grid_points : int, default=1000
        The more grid points, the

            - more detailed the explanations get and
            - the better the model performs, but
            - the slower the algorithm gets.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.isotonic import IsotonicRegression
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 2)
    >>> y = 2 * X[:, 0] - 3 * X[:, 1] + np.random.randn(100)
    >>> e = ExplainableBoostingMetaRegressor(
    ...         base_regressor=[IsotonicRegression(), IsotonicRegression(increasing=False)],
    ...         grid_points=20
    ... ).fit(X, y)
    >>> e.score(X, y)
    0.9377382292348461
    >>> e.outputs_[0] # increasing in the first feature, as it should be
    array([-4.47984456, -4.47984456, -4.47984456, -4.47984456, -3.00182713,
           -2.96627696, -1.60843287, -1.06601264, -0.92013822, -0.7217753 ,
           -0.66440783,  0.28132994,  1.33664486,  1.47592253,  1.96677286,
            2.88969439,  2.96292906,  4.33642573,  4.38506967,  6.42967225])
    >>> e.outputs_[1] # decreasing in the second feature, as it should be
    array([ 6.35605214,  6.06407947,  6.05458114,  4.8488004 ,  4.41880876,
            3.45056373,  2.64560385,  1.6138303 ,  0.89860987,  0.458301  ,
            0.33455608, -0.43609495, -1.55600464, -2.05142528, -2.42791679,
           -3.58961475, -4.80134218, -4.94421252, -5.94858712, -6.36828774])

    Notes
    -----
    Check out the original author's Github at https://github.com/interpretml/interpret and https://www.youtube.com/watch?v=MREiHgHgl0k
    for a great introduction into the operations of the algorithm.
    """

    def __init__(
        self,
        base_regressor: Any = None,
        max_rounds: int = 5000,
        learning_rate: float = 0.01,
        grid_points: int = 1000,
    ) -> None:
        """Initialize."""
        self.base_regressor = base_regressor
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.grid_points = grid_points
        self.feature_names = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None
    ) -> ExplainableBoostingMetaRegressor:
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        y : np.ndarray, 1-dimensional
            The target values.

        Returns
        -------
        ExplainableBoostingMetaRegressor
            Fitted regressor.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        X, y = check_X_y(X, y)

        sample_weight = _check_sample_weight(sample_weight, X)
        self._check_n_features(X, reset=True)

        if not isinstance(self.base_regressor, list):
            if self.base_regressor is not None:
                self.base_regressors_ = self.n_features_in_ * [self.base_regressor]
            else:
                self.base_regressors_ = self.n_features_in_ * [
                    DecisionTreeRegressor(max_depth=4)
                ]
        else:
            if len(self.base_regressor) == self.n_features_in_:
                self.base_regressors_ = self.base_regressor
            else:
                raise ValueError(
                    "Number of regressors in base_regressor should be the same as the number of features."
                )

        if self.learning_rate <= 0:
            raise ValueError("learning_rate has to be positive!")

        self.domains_ = [
            np.linspace(feature_min, feature_max, self.grid_points)
            for feature_min, feature_max in zip(X.min(axis=0), X.max(axis=0))
        ]
        self.outputs_ = [np.zeros_like(domain) for domain in self.domains_]
        self.mean_ = y.mean()

        y_copy = y.copy() - self.mean_

        self._fit(X, sample_weight, y_copy)

        self.feature_importances_ = self._feature_importances_(X)
        self.selector = self._selector(X)
        return self

    def _fit(self, X, sample_weight, y_copy):
        for i in range(self.max_rounds):
            feature_number = i % self.n_features_in_
            h = clone(self.base_regressors_[feature_number])
            x = X[:, feature_number].reshape(-1, 1)
            h.fit(x, y_copy, sample_weight=sample_weight)

            self.outputs_[feature_number] += self.learning_rate * h.predict(
                self.domains_[feature_number].reshape(-1, 1)
            )
            y_copy -= self.learning_rate * h.predict(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to get predictions of.

        Returns
        -------
        y : np.ndarray, shape (n_samples,)
            The predicted values.
        """
        X = check_array(X)
        check_is_fitted(self)
        self._check_n_features(X, reset=False)

        n = len(X)
        res = np.zeros(n)
        for feature_number in range(self.n_features_in_):
            grid = self.domains_[feature_number]
            feature_outputs = self.outputs_[feature_number][
                np.abs(
                    np.repeat(grid.reshape(-1, 1), n, axis=1) - X[:, feature_number]
                ).argmin(axis=0)
            ]
            res += feature_outputs

        return res + self.mean_

    def explain_global(self, name=None):
        """ Provides global explanation for model.
        Args:
            name: User-defined explanation name.
        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """

        lower_bound = np.inf
        upper_bound = -np.inf
        for feature_group_index in range(self.n_features_in_):
            errors = utils.fake_std(self)[feature_group_index]
            scores = self.outputs_[feature_group_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        for feature_group_index, feature_indexes in enumerate(
            [[i] for i in range(self.n_features_in_)]
        ):
            model_graph = self.outputs_[feature_group_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = utils.fake_std(self)[feature_group_index]

            if len(feature_indexes) == 1:
                # hack. remove the 0th index which is for missing values
                model_graph = model_graph[1:]
                errors = errors[1:]

                bin_labels = self.domains_[feature_indexes[0]]

                scores = list(model_graph)
                upper_bounds = list(model_graph + errors)
                lower_bounds = list(model_graph - errors)
                density_dict = {
                    "names": utils.get_fake_hist_edges(self, feature_indexes[0]),
                    "scores": utils.get_fake_hist_counts(self, feature_indexes[0]),
                }

                feature_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": scores,
                    "scores_range": bounds,
                    "upper_bounds": upper_bounds,
                    "lower_bounds": lower_bounds,
                }
                feature_list.append(feature_dict)
                density_list.append(density_dict)

                data_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": model_graph,
                    "scores_range": bounds,
                    "upper_bounds": model_graph + errors,
                    "lower_bounds": model_graph - errors,
                    "density": {
                        "names": utils.get_fake_hist_edges(self, feature_indexes[0]),
                        "scores": utils.get_fake_hist_counts(self, feature_indexes[0]),
                    },
                }

                data_dicts.append(data_dict)
            else:
                raise Exception("Interactions greater than 2 not supported.")

        overall_dict = {
            "type": "univariate",
            "names": self.__class__,
            "scores": self.feature_importances_,
        }
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_global",
                    "value": {"feature_list": feature_list},
                },
                {"explanation_type": "density", "value": {"density": density_list}},
            ],
        }

        return EBMExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=['continuous'] * self.n_features_in_,
            name=self.feature_names,
            selector=self.selector,
        )


    def _feature_importances_(self, X):
        res = []
        X_ = check_array(X)
        n = len(X_)

        for feature_number in range(self.n_features_in_):
            grid = self.domains_[feature_number]
            feature_outputs = self.outputs_[feature_number][
                np.abs(
                    np.repeat(grid.reshape(-1, 1), n, axis=1) - X_[:, feature_number]
                ).argmin(axis=0)
            ]

            res.append(feature_outputs)

        return np.mean(np.abs(res), axis=1)

    def _selector(self, X):
        if self.feature_names is None:
            self.feature_names = range(X.shape[1])

        return pd.DataFrame(
            index=range(X.shape[1]),
            data={
                'Name': feature_names,
                'Type': 'continuous',
                '# Unique': len(np.unique(X)),
                '% Non-zero': (X != 0).mean(),
            }
        )
