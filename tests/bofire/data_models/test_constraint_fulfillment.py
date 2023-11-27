from typing import List

import pandas as pd
import pytest

from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)

F = FEATURES = ["f" + str(i) for i in range(1, 11)]

C = list(range(1, 11))


def get_row(features, value: float = None, values: List[float] = None):
    if values is None:
        values = [value for _ in range(len(features))]
    return pd.DataFrame.from_dict([dict(zip(features, values))])


@pytest.mark.parametrize(
    "df, constraint, fulfilled",
    [
        (
            get_row(F[:4], 1),
            NonlinearEqualityConstraint(
                expression="f1 + f2 + f3 + f4 -4",
            ),
            True,
        ),
        (
            get_row(F[:4], 1),
            NonlinearEqualityConstraint(
                expression="f1 + f2 + f3 + f4 -3",
            ),
            False,
        ),
        (
            get_row(F[:4], 1),
            NonlinearInequalityConstraint(
                expression="f1 + f2 + f3 + f4 -5",
            ),
            True,
        ),
        (
            get_row(F[:4], 1),
            NonlinearInequalityConstraint(
                expression="f1 + f2 + f3 + f4 -2",
            ),
            False,
        ),
        (
            get_row(F[:4], 1),
            LinearEqualityConstraint(features=F[:4], coefficients=C[:4], rhs=10),
            True,
        ),
        (
            pd.concat([get_row(F[:4], 1), get_row(F[:4], 1)], ignore_index=True),
            LinearEqualityConstraint(features=F[:4], coefficients=C[:4], rhs=10),
            True,
        ),
        (
            pd.concat([get_row(F[:4], 1), get_row(F[:4], 2)]),
            LinearEqualityConstraint(features=F[:4], coefficients=C[:4], rhs=10),
            False,
        ),
        (
            get_row(F[:4], 2),
            LinearEqualityConstraint(features=F[:4], coefficients=C[:4], rhs=20),
            True,
        ),
        (
            get_row(F[:10], 1),
            LinearEqualityConstraint(features=F[:10], coefficients=C[:10], rhs=55.001),
            False,
        ),
        (
            get_row(F[:10], 1),
            LinearEqualityConstraint(features=F[:10], coefficients=C[:10], rhs=54.999),
            False,
        ),
        (
            get_row(F[:3], 1),
            LinearInequalityConstraint.from_greater_equal(
                features=F[:3], coefficients=C[:3], rhs=6
            ),
            True,
        ),
        (
            pd.concat([get_row(F[:3], 1), get_row(F[:3], 1), get_row(F[:3], 2)]),
            LinearInequalityConstraint.from_greater_equal(
                features=F[:3], coefficients=C[:3], rhs=6
            ),
            True,
        ),
        (
            pd.concat([get_row(F[:3], 1), get_row(F[:3], 0.5)]),
            LinearInequalityConstraint.from_greater_equal(
                features=F[:3], coefficients=C[:3], rhs=6
            ),
            False,
        ),
        (
            get_row(F[:3], 1),
            LinearInequalityConstraint.from_greater_equal(
                features=F[:3], coefficients=C[:3], rhs=2
            ),
            True,
        ),
        (
            get_row(F[:3], 1),
            LinearInequalityConstraint.from_greater_equal(
                features=F[:3], coefficients=C[:3], rhs=6.001
            ),
            False,
        ),
        (
            get_row(F[:3], values=[1, 1, 1]),
            NChooseKConstraint(
                features=F[:3],
                min_count=0,
                max_count=3,
                none_also_valid=True,
            ),
            True,
        ),
        (
            pd.concat(
                [get_row(F[:3], values=[1, 1, 1]), get_row(F[:3], values=[1, 1, 1])]
            ),
            NChooseKConstraint(
                features=F[:3],
                min_count=0,
                max_count=3,
                none_also_valid=True,
            ),
            True,
        ),
        (
            get_row(F[:3], values=[0, 2, 3]),
            NChooseKConstraint(
                features=F[:3],
                min_count=2,
                max_count=3,
                none_also_valid=False,
            ),
            True,
        ),
        (
            get_row(F[:3], values=[1, 2, 3]),
            NChooseKConstraint(
                features=F[:3],
                min_count=2,
                max_count=2,
                none_also_valid=False,
            ),
            False,
        ),
        (
            get_row(F[:3], values=[0, 0, 3]),
            NChooseKConstraint(
                features=F[:3],
                min_count=2,
                max_count=3,
                none_also_valid=False,
            ),
            False,
        ),
        (
            get_row(F[:3], values=[0, 0, 0]),
            NChooseKConstraint(
                features=F[:3],
                min_count=2,
                max_count=3,
                none_also_valid=False,
            ),
            False,
        ),
        (
            get_row(F[:3], values=[0, 0, 0]),
            NChooseKConstraint(
                features=F[:3],
                min_count=2,
                max_count=3,
                none_also_valid=True,
            ),
            True,
        ),
        (
            pd.concat(
                [get_row(F[:3], values=[0, 2, 3]), get_row(F[:3], values=[0, 0, 0])]
            ),
            NChooseKConstraint(
                features=F[:3],
                min_count=2,
                max_count=2,
                none_also_valid=False,
            ),
            False,
        ),
        (
            pd.concat(
                [get_row(F[:3], values=[0, 2, 3]), get_row(F[:3], values=[0, 0, 0])]
            ),
            NChooseKConstraint(
                features=F[:3],
                min_count=2,
                max_count=2,
                none_also_valid=True,
            ),
            True,
        ),
        (
            pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]}),
            InterpointEqualityConstraint(feature="a"),
            True,
        ),
        (
            pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]}),
            InterpointEqualityConstraint(feature="a"),
            False,
        ),
        (
            pd.DataFrame({"a": [1.0, 1.0, 2.0, 2.0], "b": [1.0, 2.0, 3.0, 4.0]}),
            InterpointEqualityConstraint(feature="a", multiplicity=2),
            True,
        ),
        (
            pd.DataFrame(
                {"a": [1.0, 1.0, 2.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]}
            ),
            InterpointEqualityConstraint(feature="a", multiplicity=2),
            True,
        ),
        (
            pd.DataFrame(
                {"a": [1.0, 1.0, 2.0, 3.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]}
            ),
            InterpointEqualityConstraint(feature="a", multiplicity=2),
            False,
        ),
    ],
)
def test_fulfillment(df, constraint, fulfilled):
    assert constraint.is_fulfilled(df).all() == fulfilled
