import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class PandasOneHotEncoder(OneHotEncoder):
    """
    Razor-thin layer around scikit-learn's OneHotEncoder class to return a pandas dataframe with
    the appropriate column names.
    """

    def transform(self, X):
        one_hot_encoded = super().transform(X)
        return pd.DataFrame(
            one_hot_encoded.dense() if self.sparse else one_hot_encoded,
            columns=self.get_feature_names(),
        )
