"""Input validation for estimators."""

from typing import Any, Union

import numpy as np
import pandas as pd


def check_n_features(obj: Any, X: Union[np.array, pd.DataFrame]) -> None:
    """
    Check if the number of features during fit and transformation / prediction time is the same.

    Parameters
    ----------
    obj : Any
        Some estimator.

    X : Union[np.array, pd.DataFrame]
        The input to the estimator during transformation / prediction time.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the number of features during fit and transformation / prediction time is not the same.
    """
    if X.shape[1] != obj.n_features_in_:
        raise ValueError(
            f"The dimension during fit time was {obj.n_features_in_} and now it is {X.shape[1]}. They should be the same, however."
        )
