"""Test the ImbalancedLinearRegression."""

import numpy as np
import pytest

from .. import ImbalancedLinearRegression
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
    imb = ImbalancedLinearRegression(overestimation_punishment_factor=50)
    imb.fit(X, y)

    assert (imb.predict(X) < y).mean() > 0.8


@pytest.mark.parametrize("coefs, intercept", test_batch)
def test_over_estimation(coefs, intercept):
    """Test if the model is able to overestimate."""
    X, y = _create_dataset(coefs, intercept, noise=2.0)
    imb = ImbalancedLinearRegression(overestimation_punishment_factor=0.01)
    imb.fit(X, y)

    assert (imb.predict(X) < y).mean() < 0.15
