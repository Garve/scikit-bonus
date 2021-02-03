from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List
import pandas as pd


class Dummifier(BaseEstimator, TransformerMixin):
    """
    This class is a wrapper for pd.get_dummies for use in scikit-learn pipelines.
    It one hot encodes specified columns.
    """

    def __init__(
        self, columns: Optional[List[str]] = None, drop_first: bool = False
    ) -> None:
        """
        Read all hyper parameters.

        :param columns: Columns to be one hot encoded. If None, use all columns of pandas type category or object.
        :param drop_first: Drop the first of the columns created by the one hot encoding.
        """
        self.columns = columns
        self.drop_first = drop_first

    def fit(self, X: pd.DataFrame, y: None = None) -> "Dummifier":
        """
        Does nothing.  Just for scikit-learn compatibility.

        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One hot encode the columns.

        :param X: Dataframe to be one hot encoded.
        :return: The one hot encoded dataframe.
        """
        return pd.get_dummies(X, columns=self.columns, drop_first=self.drop_first)
