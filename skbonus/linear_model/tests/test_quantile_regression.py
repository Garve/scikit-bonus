"""Test the QuantileRegression."""

import numpy as np
import pytest

from .. import QuantileRegression
from .test__general import test_batch


def _create_dataset(coefs, intercept, noise=0.0):
    np.random.seed(0)
    X = np.random.randn(1000, coefs.shape[0])
    y = X @ coefs + intercept + noise * np.random.randn(1000)

    return X, y


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_under_estimation(coefs, intercept):
    """Test if the model is able to underestimate."""
    X, y = _create_dataset(coefs, intercept, noise=2.0)
    regressor = QuantileRegression(quantile=0.1)
    regressor.fit(X, y)

    assert (regressor.predict(X) < y).mean() > 0.8


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_over_estimation(coefs, intercept):
    """Test if the model is able to overestimate."""
    X, y = _create_dataset(coefs, intercept, noise=2.0)
    regressor = QuantileRegression(quantile=0.9)
    regressor.fit(X, y)

    assert (regressor.predict(X) < y).mean() < 0.15


def test_quantile_ok():
    """Test if quantile check works."""
    X, y = np.zeros(2)
    regressor = QuantileRegression(quantile=2.0)

    with pytest.raises(ValueError):
        regressor.fit(X, y)
