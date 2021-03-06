"""Test regression metrics."""

import numpy as np
import pytest

from .. import (
    mean_absolute_deviation,
    mean_absolute_percentage_error,
    mean_directional_accuracy,
    symmetric_mean_absolute_percentage_error,
    mean_arctangent_absolute_percentage_error,
    mean_log_quotient,
)


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (np.array([1, 2, 4]), np.array([1, 1, 2]), (0 + 1 / 2 + 2 / 4) / 3),
        (np.array(100 * [5]), np.array(100 * [4]), 0.2),
        (np.array(1000 * [1]), np.array(1000 * [1]), 0.0),
        (np.array(10 * [1]), np.array(10 * [0]), 1.0),
    ],
)
def test_mape(y_true, y_pred, result):
    """Test mape."""
    np.testing.assert_almost_equal(
        mean_absolute_percentage_error(y_true, y_pred), result
    )


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (np.array([1, 2, 4]), np.array([1, 1, 2]), (0 + 1 / 3 + 2 / 6) / 3),
        (np.array(100 * [5]), np.array(100 * [4]), 1.0 / 9),
        (np.array(1000 * [1]), np.array(1000 * [1]), 0.0),
        (np.array(10 * [1]), np.array(10 * [0.0]), 1.0),
        (np.array(10 * [0.0]), np.array(10 * [0.0]), 0.0),
    ],
)
def test_smape(y_true, y_pred, result):
    """Test smape."""
    np.testing.assert_almost_equal(
        symmetric_mean_absolute_percentage_error(y_true, y_pred), result
    )


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (np.array([1, 2, 4]), np.array([1, 1, 3]), 0.5),
        (np.array([1, 5, 3, 6, 7, 8]), np.array([1, 6, 4, 4, 6, 5]), 0.6),
        (np.array(10 * [1]), np.array(10 * [1]), 1.0),
    ],
)
def test_mda(y_true, y_pred, result):
    """Test mda."""
    np.testing.assert_almost_equal(mean_directional_accuracy(y_true, y_pred), result)


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (np.array([1, 2, 4]), np.array([1, 1, 3]), 2 / 3.0),
        (np.array([1, 5, 3, 6, 7, 8]), np.array([1, 6, 4, 4, 6, 5]), 8 / 6.0),
        (np.array(10 * [1]), np.array(10 * [1]), 0.0),
    ],
)
def test_mad(y_true, y_pred, result):
    """Test mad."""
    np.testing.assert_almost_equal(mean_absolute_deviation(y_true, y_pred), result)


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (np.array([1, 2, 4]), np.array([1, 1, 3]), 0.2362087573758901),
        (
            np.array([1, 5, 3, 6, 7, 8]),
            np.array([1, 6, 4, 4, 6, 5]),
            0.2235940655863169,
        ),
        (np.array(10 * [1]), np.array(10 * [1]), 0.0),
    ],
)
def test_maape(y_true, y_pred, result):
    """Test maape."""
    np.testing.assert_almost_equal(
        mean_arctangent_absolute_percentage_error(y_true, y_pred), result
    )


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (np.array([1, 2, 4]), np.array([1, 1, 3]), 0.1877379962427844),
        (
            np.array([1, 5, 3, 6, 7, 8]),
            np.array([1, 6, 4, 4, 6, 5]),
            0.08751165372840952,
        ),
        (np.array(10 * [1]), np.array(10 * [1]), 0.0),
    ],
)
def test_mlq(y_true, y_pred, result):
    """Test mlq."""
    np.testing.assert_almost_equal(mean_log_quotient(y_true, y_pred), result)
