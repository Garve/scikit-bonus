"""Deal with dataframes."""

from typing import Any

import pandas as pd


def make_df_output(estimator: Any) -> Any:
    """
    Make a scikit-learn transformer output pandas dataframes, if its inputs are dataframes.

    Parameters
    ----------
    estimator : scikit-learn transformer
        Some transformer with a `transform` method.

    Returns
    -------
    scikit-learn transformer
        Transformer with an altered `transform` method that outputs a dataframe with the same columns and index as the input X.

    """

    class TransformerWithDataFrameOutput(estimator):
        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            cols = X.columns
            index = X.index
            out = super().transform(X)

            return pd.DataFrame(out, columns=cols, index=index)

    return TransformerWithDataFrameOutput
