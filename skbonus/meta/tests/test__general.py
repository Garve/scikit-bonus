"""Basic tests for the Meta Regressors."""

import pytest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator

from .. import ZeroInflatedRegressor, ExplainableBoostingMetaRegressor


@pytest.mark.parametrize(
    "model",
    [
        ZeroInflatedRegressor(
            classifier=RandomForestClassifier(random_state=0),
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
