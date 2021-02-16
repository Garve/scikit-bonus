from typing import Optional

import numpy as np
from sklearn.utils.validation import _check_sample_weight, check_consistent_length


def mean_absolute_deviation(
    y_true: np.array, y_pred: np.array, sample_weight: Optional[np.array] = None
) -> float:
    """
    Return the MAD (Mean Absolute Deviation) of a prediction.

    The formula is np.mean(np.abs(y_true - y_pred)).

    Parameters
    ----------
    y_true : np.array
        Observed values.

    y_pred : np.array
        Predicted values.

    sample_weight : Optional[np.array], default=None
        Individual weights for each sample.

    Returns
    -------
    float
        MAD value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 2])
    >>> mean_absolute_deviation(y_true, y_pred)
    1.0
    """
    check_consistent_length(y_true, y_pred)
    sample_weight = _check_sample_weight(sample_weight, y_true)
    return np.mean(sample_weight * np.abs(y_true - y_pred))


def mean_absolute_percentage_error(
    y_true: np.array, y_pred: np.array, sample_weight: Optional[np.array] = None
) -> float:
    """
    Return the MAPE (Mean Absolute Percentage Error) of a prediction.

    The formula is np.mean(np.abs((y_true - y_pred) / y_true)).

    Parameters
    ----------
    y_true : np.array (non-negative numbers)
        Observed values.

    y_pred : np.array
        Predicted values.

    sample_weight : Optional[np.array], default=None
        Individual weights for each sample.

    Returns
    -------
    float
        MAPE value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 2])
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.3333333333333333
    """
    check_consistent_length(y_true, y_pred)
    sample_weight = _check_sample_weight(sample_weight, y_true)
    return np.mean(sample_weight * np.abs((y_true - y_pred) / y_true))


def symmetric_mean_absolute_percentage_error(
    y_true: np.array, y_pred: np.array, sample_weight: Optional[np.array] = None
) -> float:
    """
    Return the SMAPE (Symmetric Mean Absolute Percentage Error) of a prediction.

    The formula is 2 * np.mean(np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred))).

    Parameters
    ----------
    y_true : np.array (non-negative numbers)
        Observed values.

    y_pred : np.array
        Predicted values.

    sample_weight : Optional[np.array], default=None
        Individual weights for each sample.

    Returns
    -------
    float
        SMAPE value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 2])
    >>> symmetric_mean_absolute_percentage_error(y_true, y_pred)
    0.4444444444444444
    """
    check_consistent_length(y_true, y_pred)
    sample_weight = _check_sample_weight(sample_weight, y_true)
    return 2 * np.mean(
        sample_weight * np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred))
    )


def mean_directional_accuracy(
    y_true: np.array, y_pred: np.array, sample_weight: Optional[np.array] = None
) -> float:
    """
    Return the MDA (Mean Directional Accuracy) of a prediction.

    This is the mean of the vector 1_{sgn(y_true - y_true_lag_1) = sgn(y_pred - y_true_lag_1)}.
    In plain words, it computes how often the model got the direction of the time series movement right.

    Parameters
    ----------
    y_true : np.array (non-negative numbers)
        Observed values.

    y_pred : np.array
        Predicted values.

    sample_weight : Optional[np.array], default=None
        Individual weights for each sample. The first entry is ignored since the MDA loss term consists
        of len(y_true) - 1 summands.

    Returns
    -------
    float
        MDA value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 3])
    >>> mean_directional_accuracy(y_true, y_pred)
    0.5
    """
    check_consistent_length(y_true, y_pred)
    sample_weight = _check_sample_weight(sample_weight, y_true)
    return np.mean(
        sample_weight[1:] * np.sign(np.diff(y_true))
        == np.sign(y_pred[1:] - y_true[:-1])
    )
