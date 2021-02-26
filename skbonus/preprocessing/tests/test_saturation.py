from sklearn.utils.estimator_checks import check_estimator
from ..saturation import (
    AdbudgSaturation,
    HillSaturation,
    ExponentialSaturation,
    BoxCoxSaturation,
)
import pytest


@pytest.mark.parametrize(
    "estimator",
    [BoxCoxSaturation(), AdbudgSaturation(), HillSaturation(), ExponentialSaturation()],
)
def test_check_estimator(estimator):
    check_estimator(estimator)
