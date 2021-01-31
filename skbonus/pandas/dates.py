import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, TypeVar

date_adder_type = TypeVar("date_adder_type", bound="DateAdder")


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
        dummify: bool = False,
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
        :param dummify: Create dummy variables of all columns.
        """
        self.day_of_week = day_of_week
        self.day_of_month = day_of_month
        self.day_of_year = day_of_year
        self.week_of_month = week_of_month
        self.week_of_year = week_of_year
        self.month = month
        self.year = year
        self.dummify = dummify

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

    def fit(self, X: pd.DataFrame, y: Any = None) -> date_adder_type:
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
        if self.dummify:
            return pd.get_dummies(res, columns=res.columns)
        else:
            return res
