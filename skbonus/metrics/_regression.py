from typing import Optional

import numpy as np
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils.validation import check_consistent_length


def mean_absolute_deviation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
) -> float:
    """
    Return the MAD (Mean Absolute Deviation) of a prediction.

    The formula is np.mean(np.abs(y_true - y_pred)).

    Parameters
    ----------
    y_true : np.ndarray
        Observed values.

    y_pred : np.ndarray
        Predicted values.

    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.

    multioutput : {"raw_values", "uniform_average"} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        - "raw_values": Returns a full set of errors in case of multioutput input.
        - "uniform_average": Errors of all outputs are averaged with uniform weight.

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
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    output_errors = np.average(np.abs(y_true - y_pred), weights=sample_weight)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
) -> float:
    """
    Return the MAPE (Mean Absolute Percentage Error) of a prediction.

    The formula is np.mean(np.abs((y_true - y_pred) / y_true)).

    Parameters
    ----------
    y_true : np.ndarray
        Observed values.

    y_pred : np.ndarray
        Predicted values.

    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.

    multioutput : {"raw_values", "uniform_average"} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        - "raw_values": Returns a full set of errors in case of multioutput input.
        - "uniform_average": Errors of all outputs are averaged with uniform weight.

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
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    eps = np.finfo(np.float64).eps

    output_errors = np.average(
        np.abs((y_true - y_pred) / np.maximum(y_true, eps)), weights=sample_weight
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_arctangent_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
) -> float:
    """
    Return the MAAPE (Mean Arctangent Absolute Percentage Error) of a prediction.

    The formula is np.mean(np.arctan(np.abs((y_true - y_pred) / y_true))).

    Parameters
    ----------
    y_true : np.ndarray
        Observed values.

    y_pred : np.ndarray
        Predicted values.

    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.

    multioutput : {"raw_values", "uniform_average"} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        - "raw_values": Returns a full set of errors in case of multioutput input.
        - "uniform_average": Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    float
        MAAPE value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 2])
    >>> mean_arctangent_absolute_percentage_error(y_true, y_pred)
    0.3090984060005374

    Notes
    -----
    See "A new metric of absolute percentage error for intermittent demand forecasts" by Kim & Kim.
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    eps = np.finfo(np.float64).eps

    output_errors = np.average(
        np.arctan(np.abs((y_true - y_pred) / np.maximum(y_true, eps))),
        weights=sample_weight,
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
) -> float:
    """
    Return the SMAPE (Symmetric Mean Absolute Percentage Error) of a prediction.

    The formula is np.mean(np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred))).

    Parameters
    ----------
    y_true : np.ndarray (non-negative numbers)
        Observed values.

    y_pred : np.ndarray
        Predicted values.

    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.

    multioutput : {"raw_values", "uniform_average"} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        - "raw_values": Returns a full set of errors in case of multioutput input.
        - "uniform_average": Errors of all outputs are averaged with uniform weight.

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
    0.2222222222222222
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    eps = np.finfo(np.float64).eps

    output_errors = np.average(
        np.abs((y_true - y_pred)) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps),
        weights=sample_weight,
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
) -> float:
    """
    Return the MDA (Mean Directional Accuracy) of a prediction.

    This is the mean of the vector 1_{sgn(y_true - y_true_lag_1) = sgn(y_pred - y_true_lag_1)}.
    In plain words, it computes how often the model got the direction of the time series movement right.

    Parameters
    ----------
    y_true : np.ndarray (non-negative numbers)
        Observed values.

    y_pred : np.ndarray
        Predicted values.

    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample. The first entry is ignored since the MDA loss term consists
        of len(y_true) - 1 summands.

    multioutput : {"raw_values", "uniform_average"} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        - "raw_values": Returns a full set of errors in case of multioutput input.
        - "uniform_average": Errors of all outputs are averaged with uniform weight.

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
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    output_errors = np.average(
        np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_true[:-1]),
        weights=sample_weight[1:] if sample_weight is not None else None,
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_log_quotient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
) -> float:
    """
    Return the MLQ (Mean Log Quotient) of a prediction.

    This is np.mean(np.log(y_pred / y_true)**2).

    Parameters
    ----------
    y_true : np.ndarray (non-negative numbers)
        Observed values.

    y_pred : np.ndarray
        Predicted values.

    sample_weight : Optional[np.ndarray], default=None
        Individual weights for each sample.

    multioutput : {"raw_values", "uniform_average"} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).

        - "raw_values": Returns a full set of errors in case of multioutput input.
        - "uniform_average": Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    float
        MLQ value of the input.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 2, 4])
    >>> y_pred = np.array([1, 1, 3])
    >>> mean_log_quotient(y_true, y_pred)
    0.1877379962427844

    Notes
    -----
    See Tofallis, C (2015) "A Better Measure of Relative Prediction Accuracy for Model Selection and Model Estimation",
    Journal of the Operational Research Society, 66(8),1352-1362.
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    output_errors = np.average(np.log(y_pred / y_true) ** 2, weights=sample_weight)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)
