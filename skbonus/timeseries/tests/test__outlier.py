"""Test the SpikeRemover."""

import numpy as np
from sklearn.metrics import mean_squared_error

from ..outliers import SpikeRemover


def test_spike_remover():
    """Check if the outlier reduction makes the time series be closer to the original time series without noise."""
    np.random.seed(0)

    x = np.linspace(0, 2 * np.pi, 100)

    y_clean = np.sin(x)
    y_noise = y_clean + 0.2 * np.random.randn(100)
    y_outliers = y_noise.copy()
    y_outliers[4] = 5
    y_outliers[60] = -10

    y_denoised = SpikeRemover().fit_transform(y_outliers.reshape(-1, 1))

    assert mean_squared_error(y_clean, y_denoised) < mean_squared_error(
        y_clean, y_outliers
    )
