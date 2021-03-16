"""General tests."""

import pytest
from sklearn.utils.estimator_checks import check_estimator

from ..saturation import (
    AdbudgSaturation,
    HillSaturation,
    ExponentialSaturation,
    BoxCoxSaturation,
)
from ..time import CyclicalEncoder
from ..convolution import GeneralGaussianSmoother, ExponentialDecaySmoother


@pytest.mark.parametrize(
    "estimator",
    [
        AdbudgSaturation(),
        HillSaturation(),
        ExponentialSaturation(),
        BoxCoxSaturation(),
        CyclicalEncoder(),
        ExponentialSaturation(),
        GeneralGaussianSmoother(),
    ],
)
def test_check_estimator(estimator):
    """Test if check_estimator passes."""
    check_estimator(estimator)
