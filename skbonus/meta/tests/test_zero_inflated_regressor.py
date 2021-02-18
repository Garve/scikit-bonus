"""Test the ZeroinflatedRegressor."""

from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier

from .. import ZeroInflatedRegressor


def test_check_estimator():
    z = ZeroInflatedRegressor(
        classifier=DummyClassifier(strategy="constant", constant=1),
        regressor=RandomForestRegressor(random_state=0),
    )

    check_estimator(z)
