"""Special tests for the time module."""

import numpy as np

from ..time import CyclicalEncoder


def test_cyclical_encoder():
    """Test the CyclicalEncoder."""
    ce = CyclicalEncoder()
    minutes = np.arange(60).reshape(-1, 1)
    ce_transformed = ce.fit_transform(minutes)

    np.testing.assert_almost_equal(
        ce_transformed[:7, 0],
        np.array(
            [
                1.0000000e00,
                9.9452190e-01,
                9.7814760e-01,
                9.5105652e-01,
                9.1354546e-01,
                8.6602540e-01,
                8.0901699e-01,
            ]
        ),
    )
