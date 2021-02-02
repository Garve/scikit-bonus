from skbonus.pandas.time import Dummyfier
import pandas as pd

d = Dummyfier(columns=["A", "C"], drop_first=True)
df = pd.DataFrame({"A": ["a", "b", "c", "a"], "B": [1, 2, 2, 2], "C": [0, 1, 0, 1]})
transformed = d.fit_transform(df)


def test_transform_column_names():
    assert transformed.columns.tolist() == ["B", "A_b", "A_c", "C_1"]


def test_transform_column():
    assert transformed["A_c"].values.tolist() == [0, 0, 1, 0]
