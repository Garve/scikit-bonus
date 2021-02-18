"""Test the ImbalancedLinearRegression."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from .. import ImbalancedLinearRegression

test_batch = [
    (np.array([0, 0, 3, 0, 6]), 3),
    (np.array([1, 0, -2, 0, 4, 0, -5, 0, 6]), 2),
    (np.array([4, -4]), 0),
    (np.array([0]), 0),
]


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_coefs_and_intercept__no_noise(coefs, intercept):
    """Regression problems without noise."""
    np.random.seed(0)
    X = np.random.randn(100, coefs.shape[0])
    y = X @ coefs + intercept

    imb = ImbalancedLinearRegression()
    imb.fit(X, y)
    np.testing.assert_almost_equal(coefs, imb.coef_)
    np.testing.assert_almost_equal(intercept, imb.intercept_)


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_score(coefs, intercept):
    """Tests with noise on an easy problem. A good score should be possible."""
    np.random.seed(0)
    X = np.random.randn(10000, coefs.shape[0])
    np.random.seed(0)
    y = X @ coefs + intercept + np.random.randn(10000)

    imb = ImbalancedLinearRegression()
    imb.fit(X, y)
    assert imb.score(X, y) > 0.9


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_coefs_and_intercept__no_noise_positive(coefs, intercept):
    """Test with only positive coefficients."""
    np.random.seed(0)
    X = np.random.randn(100, coefs.shape[0])
    y = X @ coefs + intercept

    imb = ImbalancedLinearRegression(positive=True)
    imb.fit(X, y)
    assert all(imb.coef_ >= 0)
    assert imb.intercept_ >= 0
    assert imb.score(X, y) > 0.3


def test_coefs_and_intercept__no_noise_sample_weight():
    """Test model with sample weights."""
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([2, 4, 6, 20, 25, 30])

    imb = ImbalancedLinearRegression()
    imb.fit(X, y, sample_weight=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    np.testing.assert_almost_equal(5, imb.coef_[0])


def test_coefs_and_intercept__no_noise_regularization():
    """Test model with sample weights."""
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([2, 4, 6, 8, 10, 12])

    imbs = [
        ImbalancedLinearRegression(alpha=alpha, l1_ratio=0.5).fit(X, y)
        for alpha in range(5)
    ]
    coefs = np.array([imb.coef_[0] for imb in imbs])
    np.testing.assert_almost_equal(2, coefs[0])
    for i in range(4):
        assert coefs[i] > coefs[i + 1]


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_under_estimation(coefs, intercept):
    """Test if the model is able to underestimate."""
    np.random.seed(0)
    X = np.random.randn(100, coefs.shape[0])
    y = X @ coefs + intercept + 2 * np.random.randn(100)

    imb = ImbalancedLinearRegression(overestimation_punishment_factor=50)
    imb.fit(X, y)
    assert (imb.predict(X) < y).mean() > 0.8


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_over_estimation(coefs, intercept):
    """Test if the model is able to overestimate."""
    np.random.seed(0)
    X = np.random.randn(100, coefs.shape[0])
    y = X @ coefs + intercept + 2 * np.random.randn(100)

    imb = ImbalancedLinearRegression(overestimation_punishment_factor=0.01)
    imb.fit(X, y)
    assert (imb.predict(X) < y).mean() < 0.15


def test_check_estimator():
    l = ImbalancedLinearRegression()
    check_estimator(l)
