"""General tests."""

import pytest
from sklearn.utils.estimator_checks import check_estimator

from ..naive import QuantileBoxEnvelope


@pytest.mark.parametrize(
    "estimator",
    [
        QuantileBoxEnvelope()
    ],
)
def test_check_estimator(estimator):
    """Test if check_estimator passes."""
    check_estimator(estimator)
