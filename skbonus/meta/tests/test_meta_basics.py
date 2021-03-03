"""Basic tests for the Meta Regressors."""

import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from .. import ZeroInflatedRegressor, ExplainableBoostingMetaRegressor


@pytest.mark.parametrize(
    "model",
    [
        ZeroInflatedRegressor(
            regressor=RandomForestRegressor(random_state=0),
        ),
        ExplainableBoostingMetaRegressor(
            base_regressor=DecisionTreeRegressor(), max_rounds=500
        ),
    ],
)
def test_check_estimator(model):
    """Test if check_estimator works."""
    check_estimator(model)
