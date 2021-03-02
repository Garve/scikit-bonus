"""Deal with dataframes."""

import pandas as pd


def make_df_output(estimator):
    class EstimatorWithDataFrameOutput(estimator):
        def transform(self, X):
            cols = X.columns
            index = X.index
            out = super().transform(X)
            return pd.DataFrame(out, columns=cols, index=index)

    return EstimatorWithDataFrameOutput
