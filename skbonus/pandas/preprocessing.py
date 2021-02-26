"""Preprocess data for training with a focus on pandas compatibility."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder as ScikitLearnOneHotEncoder
from sklearn.utils.validation import check_is_fitted, check_array


class OneHotEncoderWithNames(ScikitLearnOneHotEncoder):
    """
    Razor-thin layer around scikit-learn's OneHotEncoder class to return a pandas dataframe with the appropriate column names.

    Description from the maintainers of scikit-learn:

    Encode categorical features as a one-hot numeric array.
    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse``
    parameter).

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

    drop : {'first', 'if_binary'} or a array-like of shape (n_features,), default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into a neural network or an unregularized regression.
        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - 'if_binary' : drop the first category in each feature with two
          categories. Features with 1 or more than 2 categories are
          left intact.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped.

    sparse : bool, default=True
        Will return sparse matrix if set True else will return an array.

    dtype : number type, default=float
        Desired dtype of output.

    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).

    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
          to be dropped for each feature.
        - ``drop_idx_[i] = None`` if no category is to be dropped from the
          feature with index ``i``, e.g. when `drop='if_binary'` and the
          feature isn't binary.
        - ``drop_idx_ = None`` if all the transformed features will be
          retained.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 1], 'B': ['a', 'b', 'c']})
    >>> OneHotEncoderWithNames().fit_transform(df)
       A_1  A_2  B_a  B_b  B_c
    0    1    0    1    0    0
    1    0    1    0    1    0
    2    1    0    0    0    1
    """

    def fit(self, X: pd.DataFrame, y: None = None):
        """
        Fits a OneHotEncoder while also storing the dataframe column names that let us check if the columns match when calling the transform method.

        Parameters
        ----------
        X : pd.DataFrame
            Fit the OneHotEncoder on this dataframe.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        OneHotEncoderWithNames
            Fitted transformer.
        """
        self.column_names_ = X.columns
        self._check_n_features(X, reset=True)

        return super().fit(X, y)

    def _replace_prefix(self, ohe_column_name):
        feature_name, feature_value = ohe_column_name.split("_", 1)
        feature_number = int(feature_name[1:])

        return "_".join([self.column_names_[feature_number], feature_value])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One hot encode the input dataframe.

        Parameters
        ----------
        X : pd.DataFrame
            Input to be one hot encoded. The column names should be the same as during the fit method,
            including the same order.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe containing the one hot encoded data and proper column names.

        Raises
        ------
        AssertionError
            If the column names during training and transformation time are not identical.
        """
        if X.columns.tolist() != self.column_names_.tolist():
            raise AssertionError(
                "Column names during fit and transform time should be identical, including the order."
            )

        one_hot_encoded = super().transform(X)
        feature_names = [self._replace_prefix(x) for x in self.get_feature_names()]

        return pd.DataFrame(
            one_hot_encoded.todense() if self.sparse else one_hot_encoded,
            columns=feature_names,
        ).astype(int)


class DateTimeExploder(BaseEstimator, TransformerMixin):
    """
    Transform a pandas dataframe with columns (*, start_date, end_date) into a longer format with columns (*, date).

    This is useful if you deal with datasets that contain special time periods per row, but you need a single date per row.
    See the examples for more details.

    Parameters
    ----------
    name : str
        Name of the new output date column.

    start_column : str
        Start date of the period.

    end_column : str
        End date of the period.

    frequency : str
        A pandas time frequency. Can take values like "d" for day or "m" for month. A full list can
        be found on https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases.
        If None, the transformer tries to infer it during fit time.

    drop : bool, default=True
        Whether to drop the `start_column` and `end_column` in the transformed output.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    "Data": ["a", "b", "c"],
    ...    "Start": pd.date_range("2020-01-01", periods=3),
    ...    "End": pd.date_range("2020-01-03", periods=3)
    ... })
    >>> df
      Data      Start        End
    0    a 2020-01-01 2020-01-03
    1    b 2020-01-02 2020-01-04
    2    c 2020-01-03 2020-01-05

    >>> DateTimeExploder(name="output_date", start_column="Start", end_column="End", frequency="d").fit_transform(df)
      Data output_date
    0    a  2020-01-01
    0    a  2020-01-02
    0    a  2020-01-03
    1    b  2020-01-02
    1    b  2020-01-03
    1    b  2020-01-04
    2    c  2020-01-03
    2    c  2020-01-04
    2    c  2020-01-05

    """

    def __init__(
        self,
        name: str,
        start_column: str,
        end_column: str,
        frequency: str,
        drop: bool = True,
    ) -> None:
        """Initialize."""
        self.name = name
        self.start_column = start_column
        self.end_column = end_column
        self.frequency = frequency
        self.drop = drop

    def fit(self, X: pd.DataFrame, y=None) -> "DateTimeExploder":
        """
        Fits the estimator.

        In this special case, nothing is done.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        DateTimeExploder
            Fitted transformer.
        """
        self._check_n_features(X, reset=True)

        return self

    def _start_and_end_to_date_range(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DatetimeIndex:
        """
        Given a start and an end date, output a continuous DatetimeIndex.

        Parameters
        ----------
        start : pd.Timestamp
            Start date of the period.

        end : pd.Timestamp
            End date of the period.

        Returns
        -------
        pd.DatetimeIndex
            A date range from `start` to `end` with a frequency of `self.frequency`.
        """
        return pd.date_range(start=start, end=end, freq=self.frequency)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas dataframe with the columns `self.start_column` and `end_column` containing dates.

        Returns
        -------
        pd.DataFrame
            A longer dataframe with one date per row.
        """
        check_is_fitted(self)
        check_array(X, dtype=None)
        self._check_n_features(X, reset=False)

        return (
            X.assign(
                **{
                    self.name: lambda df: df.apply(
                        lambda row: self._start_and_end_to_date_range(
                            start=row[self.start_column],
                            end=row[self.end_column],
                        ),
                        axis=1,
                    )
                }
            )
            .explode(self.name)
            .drop(columns=self.drop * ["Start", "End"])
        )
