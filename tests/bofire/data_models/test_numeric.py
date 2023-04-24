import numpy as np
import pandas as pd
import pytest

from bofire.data_models.domain.domain import is_numeric


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            pd.DataFrame(
                {
                    "col": [1, 2, 10, np.nan, "a"],
                    "col2": ["a", 10, 30, 40, 50],
                    "col3": [1, 2, 3, 4, 5.0],
                }
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "col": [1, 2, 10, np.nan, 6],
                    "col2": [5, 10, 30, 40, 50],
                    "col3": [1, 2, 3, 4, 5.0],
                }
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "col": [1, 2, 10, 7.0, 6],
                    "col2": [5, 10, 30, 40, 50],
                    "col3": [1, 2, 3, 4, 5.0],
                }
            ),
            True,
        ),
        (
            pd.Series([1, 2, 10, 7.0, 6]),
            True,
        ),
        (
            pd.Series([1, 2, "abc", 7.0, 6]),
            False,
        ),
    ],
)
def test_is_numeric(df, expected):
    assert is_numeric(df) == expected
