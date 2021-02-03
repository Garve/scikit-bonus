import numpy as np
from sklearn.utils.validation import check_consistent_length


def mape(y_true: np.array, y_pred: np.array) -> float:
    """
    Returns the MAPE (Mean Absolute Percentage Error) of a prediction, i.e. the average of the vector
    |(y_true - y_pred) / y_true|.

    :param y_true: True, observed values.
    :param y_pred: Predicted values.
    :return: The MAPE of the inputs.
    """
    check_consistent_length(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def smape(y_true: np.array, y_pred: np.array) -> float:
    """
    Returns the SMAPE (Symmetric Mean Absolute Percentage Error) of a prediction, i.e. the average of the vector
    2 * |(y_true - y_pred)| / (|y_true| + |y_pred|).

    :param y_true: True, observed values.
    :param y_pred: Predicted values.
    :return: The SMAPE of the inputs.
    """
    check_consistent_length(y_true, y_pred)
    return 2 * np.mean(np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred)))


def mda(y_true: np.array, y_pred: np.array) -> float:
    """
    Returns the MDA (Mean Directional Accuracy) of a prediction, i.e. the average of the vector
    1_{sgn(y_true - y_true_lag_1) = sgn(y_pred - y_true_lag_1)}. In plain words, it computes how often
    the model got the direction of the time series movement right.

    :param y_true: True, observed values.
    :param y_pred: Predicted values.
    :return: The MDA of the inputs.
    """
    check_consistent_length(y_true, y_pred)
    return np.mean(np.sign(np.diff(y_true)) == np.sign(y_pred[1:] - y_true[:-1]))
