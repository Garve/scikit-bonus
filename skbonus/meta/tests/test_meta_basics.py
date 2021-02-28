"""Basic tests for the Meta Regressors."""

import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyClassifier

from .. import ZeroInflatedRegressor, ExplainableBoostingMetaRegressor


@pytest.mark.parametrize(
    "model",
    [
        ZeroInflatedRegressor(
            classifier=DummyClassifier(strategy="constant", constant=1),
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
