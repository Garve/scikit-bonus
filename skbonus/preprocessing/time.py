from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
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
            Used for inferring te ranges of the data, if not provided during initialization.

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
