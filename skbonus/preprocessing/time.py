from typing import List, Tuple, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from skbonus.utils.validation import check_n_features


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Break each cyclic feature into two new features, corresponding to the representation of this feature on a circle.

    For example, take the hours from 0 to 23. On a normal, round  analog clock,
    these features are perfectly aligned on a circle already. You can do the same with days, month, ...

    The column names affected by default are

        - second
        - minute
        - hour
        - day_of_week
        - day_of_month
        - day_of_year
        - week_of_month
        - week_of_year
        - month

    You can add more with the additional_cycles parameter.

    Notes
    -----
    This method has the advantage that close points in time stay close together. See the examples below.

    Otherwise, if algorithms deal with the raw value for hour they cannot know that 0 and 23 are actually close.
    Another possibility is one hot encoding the hour. This has the disadvantage that it breaks the distances
    between different hours. Hour 5 and 16 have the same distance as hour 0 and 23 when doing this.

    Parameters
    ----------
    additional_cycles : Optional[Dict[str, Dict[str, int]]], default=None
        Define additional additional_cycles in the form {cycle_name: {"min": min_value, "max": max_value}}, e.g.
        {"day_of_week": {"min": 1, "max": 7}}. Probably you need this only for very specific additional_cycles, as
        these ones are already implemented:

            - "second": {"min": 0, "max": 59},
            - "minute": {"min": 0, "max": 59},
            - "hour": {"min": 0, "max": 23},
            - "day_of_week": {"min": 1, "max": 7},
            - "day_of_month": {"min": 1, "max": 31},
            - "day_of_year": {"min": 1, "max": 366},
            - "week_of_month": {"min": 1, "max": 5},
            - "week_of_year": {"min": 1, "max": 53},
            - "month": {"min": 1, "max": 12}

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

    def fit(self, X: np.array, y=None) -> "CyclicalEncoder":
        """
        Fit the estimator. In this special case, nothing is done.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        CyclicalEncoder
            Fitted transformer.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        if self.cycles is None:
            self.cycles_ = list(zip(X.min(axis=0), X.max(axis=0)))
        else:
            self.cycles_ = self.cycles

        return self

    def transform(self, X: np.array) -> np.array:
        """
        Add the cyclic features to the dataframe.

        Parameters
        ----------
        X : np.array
            The data with cyclical features in the columns.

        Returns
        -------
        np.array
            The encoded data with twice as man columns as the original.
        """
        check_is_fitted(self)
        X = check_array(X)
        check_n_features(self, X)

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
