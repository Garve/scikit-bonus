"""Test the LADRegressor."""

import numpy as np
import pytest

from skbonus.linear_model import LADRegression


@pytest.mark.parametrize(
    "coefs, intercept",
    [
        (np.array([0, 0, 3, 0, 6]), 3),
        (np.array([1, 0, -2, 0, 4, 0, -5, 0, 6]), 2),
        (np.array([4, -4]), 0),
        (np.array([0]), 0),
    ],
)
def test_coefs_and_intercept__no_noise(coefs, intercept):
    """Regression problems without noise."""
    np.random.seed(0)
    X = np.random.randn(100, coefs.shape[0])
    y = X @ coefs + intercept

    lad = LADRegression()
    lad.fit(X, y)
    np.testing.assert_almost_equal(coefs, lad.coef_)
    np.testing.assert_almost_equal(intercept, lad.intercept_)


@pytest.mark.parametrize(
    "coefs, intercept",
    [
        (np.array([0, 0, 3, 0, 6]), 3),
        (np.array([1, 0, -2, 0, 4, 0, -5, 0, 6]), 2),
        (np.array([4, -4]), 0),
        (np.array([0]), 0),
    ],
)
def test_score(coefs, intercept):
    """Tests with noise on an easy problem. A good score should be possible."""
    np.random.seed(0)
    X = np.random.randn(10000, coefs.shape[0])
    np.random.seed(0)
    y = X @ coefs + intercept + np.random.randn(10000)

    lad = LADRegression()
    lad.fit(X, y)
    assert lad.score(X, y) > 0.9


@pytest.mark.parametrize(
    "coefs, intercept",
    [
        (np.array([0, 0, 3, 0, 6]), 3),
        (np.array([1, 0, -2, 0, 4, 0, -5, 0, 6]), 2),
        (np.array([4, -4]), 0),
        (np.array([0]), 0),
    ],
)
def test_coefs_and_intercept__no_noise_positive(coefs, intercept):
    """Test with only positive coefficients."""
    np.random.seed(0)
    X = np.random.randn(100, coefs.shape[0])
    y = X @ coefs + intercept

    lad = LADRegression(positive=True)
    lad.fit(X, y)
    assert all(lad.coef_ >= 0)
    assert lad.intercept_ >= 0
    assert lad.score(X, y) > 0.3


def test_coefs_and_intercept__no_noise_sample_weight():
    """Test model with sample weights."""
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([2, 4, 6, 20, 25, 30])

    lad = LADRegression()
    lad.fit(X, y, sample_weight=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(5, lad.coef_[0], decimal=5)
