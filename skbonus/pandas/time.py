import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Optional, List


class DateFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    This class enriches pandas dataframes with a time index with new columns.
    This is especially useful when dealing with time series regressions or classifications.
    """

    def __init__(
        self,
        day_of_week: bool = False,
        day_of_month: bool = True,
        day_of_year: bool = False,
        week_of_month: bool = False,
        week_of_year: bool = False,
        month: bool = True,
        year: bool = True,
    ) -> None:
        """
        Read all hyper parameters.

        :param day_of_week: Add day of week.
        :param day_of_month: Add day of month.
        :param day_of_year: Add day of year.
        :param week_of_month: Add week of month.
        :param week_of_year: Add week of year (ISO).
        :param month: Add month.
        :param year: Add year.
        """
        self.day_of_week = day_of_week
        self.day_of_month = day_of_month
        self.day_of_year = day_of_year
        self.week_of_month = week_of_month
        self.week_of_year = week_of_year
        self.month = month
        self.year = year

    @staticmethod
    def _add_day_of_week(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        """
        Adds the day of week of the index of a dataframe as a new column into this dataframe.

        :param X: A dataframe with a date index.
        :param use: Only append the feature if use == True, else pass.
        :return: A dataframe with another column containing the day of week of the index.
        """
        return X.assign(day_of_week=lambda df: df.index.weekday) if use else X

    @staticmethod
    def _add_day_of_month(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        """
        Adds the day of month of the index of a dataframe as a new column into this dataframe.

        :param X: A dataframe with a date index.
        :param use: Only append the feature if use == True, else pass.
        :return: A dataframe with another column containing the day of month of the index.
        """
        return X.assign(day_of_month=lambda df: df.index.day) if use else X

    @staticmethod
    def _add_day_of_year(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        """
        Adds the day of year of the index of a dataframe as a new column into this dataframe.

        :param X: A dataframe with a date index.
        :param use: Only append the feature if use == True, else pass.
        :return: A dataframe with another column containing the day of year of the index.
        """
        return X.assign(day_of_year=lambda df: df.index.dayofyear) if use else X

    @staticmethod
    def _add_week_of_month(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        """
        Adds the week of month of the index of a dataframe as a new column into this dataframe.

        :param X: A dataframe with a date index.
        :param use: Only append the feature if use == True, else pass.
        :return: A dataframe with another column containing the week of month of the index.
        """
        return (
            X.assign(week_of_month=lambda df: np.ceil(df.index.day / 7).astype(int))
            if use
            else X
        )

    @staticmethod
    def _add_week_of_year(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        """
        Adds the week of year of the index of a dataframe as a new column into this dataframe.

        :param X: A dataframe with a date index.
        :param use: Only append the feature if use == True, else pass.
        :return: A dataframe with another column containing the week of year of the index.
        """
        return X.assign(week_of_year=lambda df: df.index.weekofyear) if use else X

    @staticmethod
    def _add_month(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        """
        Adds the month of the index of a dataframe as a new column into this dataframe.

        :param X: A dataframe with a date index.
        :param use: Only append the feature if use == True, else pass.
        :return: A dataframe with another column containing the month of the index.
        """
        return X.assign(month=lambda df: df.index.month) if use else X

    @staticmethod
    def _add_year(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        """
        Adds the year of the index of a dataframe as a new column into this dataframe.

        :param X: A dataframe with a date index.
        :param use: Only append the feature if use == True, else pass.
        :return: A dataframe with another column containing the year of the index.
        """
        return X.assign(year=lambda df: df.index.year) if use else X

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DateFeaturesAdder":
        """
        Does nothing. Just for scikit-learn compatibility.
        :param X:
        :param y:
        :return:
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add numerous new time features to the dataframe X.

        :param X: Input dataframe with a time index.
        :return: A dataframe with numerous new columns containing time features.
        """
        res = (
            X.pipe(self._add_day_of_week, use=self.day_of_week)
            .pipe(self._add_day_of_month, use=self.day_of_month)
            .pipe(self._add_day_of_year, use=self.day_of_year)
            .pipe(self._add_week_of_month, use=self.week_of_month)
            .pipe(self._add_week_of_year, use=self.week_of_year)
            .pipe(self._add_month, use=self.month)
            .pipe(self._add_year, use=self.year)
        )
        return res


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


class PowerTrendAdder(BaseEstimator, TransformerMixin):
    """
    Adds a power trend to the data.
    """

    def __init__(self, power: float = 1) -> None:
        """
        Read all hyper parameters.

        :param power: Exponent to use for the trend, i.e. linear (power=1), root (power=0.5), or cube (power=3).
        """
        self.power = power

    def fit(self, X: pd.DataFrame, y: None = None) -> "PowerTrendAdder":
        """
        Fits the model. It assigns value 0 to the first item of the time index and 1 to the second one.
        This way, we can get a value for any other date in a linear fashion.

        :param X: Input dataframe with a time index.
        :param y:
        :return:
        """
        t1 = X.index[0].value
        t2 = X.index[1].value

        self.mapper_ = lambda x: (x - t1) / (t2 - t1)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add the trend to the input dataframe.

        :param X: Input dataframe with a time index.
        :return: Dataframe with an additional trend columns.
        """
        index = X.index.astype(int)

        return X.assign(trend=self.mapper_(index) ** self.power)
