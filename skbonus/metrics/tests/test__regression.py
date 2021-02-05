import numpy as np
import pytest

from skbonus.metrics import mape, smape, mda


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
    np.testing.assert_almost_equal(mape(y_true, y_pred), result)


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (np.array([1, 2, 4]), np.array([1, 1, 2]), 2 * (0 + 1 / 3 + 2 / 6) / 3),
        (np.array(100 * [5]), np.array(100 * [4]), 2.0 / 9),
        (np.array(1000 * [1]), np.array(1000 * [1]), 0.0),
        (np.array(10 * [1]), np.array(10 * [0.0]), 2.0),
    ],
)
def test_smape(y_true, y_pred, result):
    np.testing.assert_almost_equal(smape(y_true, y_pred), result)


@pytest.mark.parametrize(
    "y_true, y_pred, result",
    [
        (np.array([1, 2, 4]), np.array([1, 1, 3]), 0.5),
        (np.array([1, 5, 3, 6, 7, 8]), np.array([1, 6, 4, 4, 6, 5]), 0.6),
        (np.array(10 * [1]), np.array(10 * [1]), 1.0),
    ],
)
def test_mda(y_true, y_pred, result):
    np.testing.assert_almost_equal(mda(y_true, y_pred), result)
