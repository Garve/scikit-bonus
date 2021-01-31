import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, TypeVar

date_adder_type = TypeVar("date_adder_type", bound="DateAdder")


class DateAdder(BaseEstimator, TransformerMixin):
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
        return X.assign(day_of_week=lambda df: df.index.weekday) if use else X

    @staticmethod
    def _add_day_of_month(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        return X.assign(day_of_month=lambda df: df.index.day) if use else X

    @staticmethod
    def _add_day_of_year(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        return X.assign(day_of_year=lambda df: df.index.dayofyear) if use else X

    @staticmethod
    def _add_week_of_month(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        return (
            X.assign(week_of_month=lambda df: np.ceil(df.index.day / 7).astype(int))
            if use
            else X
        )

    @staticmethod
    def _add_week_of_year(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        return X.assign(week_of_year=lambda df: df.index.weekofyear) if use else X

    @staticmethod
    def _add_month(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        return X.assign(month=lambda df: df.index.month) if use else X

    @staticmethod
    def _add_year(X: pd.DataFrame, use: bool) -> pd.DataFrame:
        return X.assign(year=lambda df: df.index.year) if use else X

    def fit(self, X: pd.DataFrame, y: Any = None) -> date_adder_type:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
