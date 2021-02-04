import numpy as np
from sklearn.utils.validation import check_consistent_length


def mape(y_true: np.array, y_pred: np.array) -> float:
    """
    Returns the MAPE (Mean Absolute Percentage Error) of a prediction, i.e. the average of the vector
    |(y_true - y_pred) / y_true|.

    Parameters
    ----------
    y_true : np.array (non-negative numbers)
        Observed values.
    y_pred : np.array
        Predicted values.

    Returns
    -------
    mape : float
        MAPE value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 2])
    >>> mape(y_true, y_pred)
    0.3333333333333333
    """
    check_consistent_length(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def smape(y_true: np.array, y_pred: np.array) -> float:
    """
    Returns the SMAPE (Symmetric Mean Absolute Percentage Error) of a prediction, i.e. the average of the vector
    2 * |(y_true - y_pred)| / (|y_true| + |y_pred|).

    Parameters
    ----------
    y_true : np.array (non-negative numbers)
        Observed values.
    y_pred : np.array
        Predicted values.

    Returns
    -------
    smape : float
        SMAPE value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 2])
    >>> smape(y_true, y_pred)
    0.4444444444444444
    """
    check_consistent_length(y_true, y_pred)
    return 2 * np.mean(np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred)))


def mda(y_true: np.array, y_pred: np.array) -> float:
    """
    Returns the MDA (Mean Directional Accuracy) of a prediction, i.e. the average of the vector
    1_{sgn(y_true - y_true_lag_1) = sgn(y_pred - y_true_lag_1)}. In plain words, it computes how often
    the model got the direction of the time series movement right.

    Parameters
    ----------
    y_true : np.array (non-negative numbers)
        Observed values.
    y_pred : np.array
        Predicted values.

    Returns
    -------
    mda : float
        MDA value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 3])
    >>> mda(y_true, y_pred)
    0.5
    """
    check_consistent_length(y_true, y_pred)
    return np.mean(np.sign(np.diff(y_true)) == np.sign(y_pred[1:] - y_true[:-1]))


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
