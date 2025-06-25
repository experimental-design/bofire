from math import nan

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pydantic.error_wrappers import ValidationError
from pytest import fixture

from bofire.data_models.api import Outputs
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.domain.constraints import Constraints
from bofire.data_models.domain.domain import is_numeric
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    Feature,
)
from bofire.data_models.objectives.api import TargetObjective
from bofire.utils.subdomain import get_subdomain


def test_empty_domain():
    assert Domain() == Domain(
        inputs=Inputs(),
        outputs=Outputs(),
        constraints=Constraints(),
    )


obj = TargetObjective(target_value=1, steepness=2, tolerance=3, w=0.5)

if_ = CategoricalInput(key="if", categories=["a", "b"])
of_ = ContinuousOutput(key="of", objective=obj)
if1 = ContinuousInput(key="f1", bounds=(0, 10))
if1_ = ContinuousInput(key="f1", bounds=(0, 1))
if2 = ContinuousInput(key="f2", bounds=(0, 10))
of1 = ContinuousOutput(key="f1", objective=obj)
of1_ = ContinuousOutput(key="f1", objective=obj)
of2 = ContinuousOutput(key="f2", objective=obj)
of3 = ContinuousOutput(key="of3", objective=obj)


@fixture
def input_list():
    return [
        ContinuousInput(key="ci1", bounds=(0, 1)),
        ContinuousInput(key="ci2", bounds=(0, 1)),
    ]


@fixture
def output_list():
    objective = TargetObjective(target_value=1, steepness=2, tolerance=3, w=0.5)
    return [
        ContinuousOutput(key="co1", objective=objective),
        ContinuousOutput(key="co2", objective=objective),
    ]


@fixture
def constraint_list(input_list):
    return [
        LinearEqualityConstraint(
            features=[inp.key for inp in input_list],
            coefficients=[1.0] * len(input_list),
            rhs=11,
        ),
    ]


def test_from_lists(input_list, output_list, constraint_list):
    assert Domain.from_lists(
        inputs=input_list,
        outputs=output_list,
        constraints=constraint_list,
    ) == Domain(
        inputs=Inputs(features=input_list),
        outputs=Outputs(features=output_list),
        constraints=Constraints(constraints=constraint_list),
    )


@pytest.mark.parametrize(
    "inputs, outputs, constraints",
    [
        (
            [if1, if2],
            [of3],
            [
                NChooseKConstraint(
                    features=["f1", "f2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        ),
        (
            [if1, if2],
            [],
            [
                NChooseKConstraint(
                    features=["f1", "f2"],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
            ],
        ),
    ],
)
def test_valid_constraints_in_domain(outputs, inputs, constraints):
    Domain(
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
    )


@pytest.mark.parametrize(
    "inputs, outputs, constraints",
    [
        (
            [],
            [],
            [
                NChooseKConstraint(
                    features=["x", "t55"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        ),
        (
            [if1],
            [],
            [
                NChooseKConstraint(
                    features=["f1", "x"],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
            ],
        ),
        (
            [],
            [of1],
            [
                NChooseKConstraint(
                    features=["f1", "x"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        ),
        (
            [if1],
            [of1],
            [
                NChooseKConstraint(
                    features=["f1", "f2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        ),
    ],
)
def test_unknown_features_in_domain(outputs, inputs, constraints):
    with pytest.raises(ValidationError):
        Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=constraints,
        )


data = pd.DataFrame.from_dict(
    {
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "out1": [nan, 1.0, 2.0, 3.0, nan, nan],
        "out2": [nan, 1.0, 2.0, 3.0, 4.0, 5.0],
        "valid_out1": [1, 0, 1, 1, 1, 1],
        "valid_out2": [1, 1, 0, 1, 1, 0],
    },
)

if1 = ContinuousInput(key="x1", bounds=(1, 10))
if2 = ContinuousInput(key="x2", bounds=(1, 10))

of1 = ContinuousOutput(key="out1", objective=obj)
of2 = ContinuousOutput(key="out2", objective=obj)

c1 = LinearEqualityConstraint(features=["x1", "x2"], coefficients=[5, 5], rhs=15.0)

of1_ = ContinuousOutput(key="out3", objective=None)
of2_ = ContinuousOutput(key="out4", objective=None)

domain = Domain(
    inputs=Inputs(features=[if1, if2]),
    outputs=Outputs(features=[of1, of2]),
)
domain2 = Domain(
    inputs=Inputs(features=[if1, if2]),
    outputs=Outputs(features=[of1, of2]),
    constraints=Constraints(constraints=[c1]),
)


def test_coerce_invalids():
    domain = Domain(
        inputs=Inputs(features=[if1, if2]),
        outputs=Outputs(features=[of1, of2]),
    )
    experiments = domain.coerce_invalids(data)
    expected = pd.DataFrame.from_dict(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "out1": [nan, nan, 2.0, 3.0, nan, nan],
            "out2": [nan, 1.0, nan, 3.0, 4.0, nan],
            "valid_out1": [1, 0, 1, 1, 1, 1],
            "valid_out2": [1, 1, 0, 1, 1, 0],
        },
    )
    assert_frame_equal(experiments, expected, check_dtype=False)


@pytest.mark.parametrize("method", ["mean", "median"])
def test_aggregate_by_duplicates_no_continuous(method):
    full = pd.DataFrame.from_dict(
        {
            "x1": ["a", "b", "c", "a"],
            "x2": ["b", "b", "c", "b"],
            "out1": [4.0, 5.0, 6.0, 3.0],
            "out2": [-4.0, -5.0, -6.0, -3.0],
            "valid_out1": [1, 1, 1, 1],
            "valid_out2": [1, 1, 1, 1],
        },
    )
    expected_aggregated = pd.DataFrame.from_dict(
        {
            "labcode": ["1-4", "2", "3"],
            "x1": ["a", "b", "c"],
            "x2": ["b", "b", "c"],
            "out1": [3.5, 5.0, 6.0],
            "out2": [-3.5, -5.0, -6.0],
            "valid_out1": [1, 1, 1],
            "valid_out2": [1, 1, 1],
        },
    )
    domain = Domain(
        inputs=Inputs(
            features=[
                CategoricalInput(key="x1", categories=["a", "b", "c"]),
                CategoricalInput(key="x2", categories=["a", "b", "c"]),
            ],
        ),
        outputs=Outputs(features=[of1, of2]),
    )
    aggregated, duplicated_labcodes = domain.aggregate_by_duplicates(
        full,
        prec=2,
        method=method,
    )
    assert duplicated_labcodes == [["1", "4"]]
    assert_frame_equal(
        aggregated,
        expected_aggregated,
        check_dtype=False,
        check_like=True,
    )


@pytest.mark.parametrize("method", ["mean", "median"])
def test_aggregate_by_duplicates(method):
    # dataframe with duplicates
    full = pd.DataFrame.from_dict(
        {
            "x1": [1.0, 2.0, 3.0, 1.0],
            "x2": [1.0, 2.0, 3.0, 1.0],
            "out1": [4.0, 5.0, 6.0, 3.0],
            "out2": [-4.0, -5.0, -6.0, -3.0],
            "valid_out1": [1, 1, 1, 1],
            "valid_out2": [1, 1, 1, 1],
        },
    )
    expected_aggregated = pd.DataFrame.from_dict(
        {
            "labcode": ["1-4", "2", "3"],
            "x1": [1.0, 2.0, 3.0],
            "x2": [1.0, 2.0, 3.0],
            "out1": [3.5, 5.0, 6.0],
            "out2": [-3.5, -5.0, -6.0],
            "valid_out1": [1, 1, 1],
            "valid_out2": [1, 1, 1],
        },
    )
    domain = Domain(
        inputs=Inputs(features=[if1, if2]),
        outputs=Outputs(features=[of1, of2]),
    )
    aggregated, duplicated_labcodes = domain.aggregate_by_duplicates(
        full,
        prec=2,
        method=method,
    )
    assert duplicated_labcodes == [["1", "4"]]
    assert_frame_equal(
        aggregated,
        expected_aggregated,
        check_dtype=False,
        check_like=True,
    )
    # dataset without duplicates
    full = pd.DataFrame.from_dict(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [1.0, 2.0, 3.0, 4.0],
            "out1": [4.0, 5.0, 6.0, 3.0],
            "out2": [-4.0, -5.0, -6.0, -3.0],
            "valid_out1": [1, 1, 1, 1],
            "valid_out2": [1, 1, 1, 1],
        },
    )
    expected_aggregated = pd.DataFrame.from_dict(
        {
            "labcode": ["0", "1", "2", "3"],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [1.0, 2.0, 3.0, 4.0],
            "out1": [4.0, 5.0, 6.0, 3.0],
            "out2": [-4.0, -5.0, -6.0, -3.0],
            "valid_out1": [1, 1, 1, 1],
            "valid_out2": [1, 1, 1, 1],
        },
    )
    domain = Domain(
        inputs=Inputs(features=[if1, if2]),
        outputs=Outputs(features=[of1, of2]),
    )
    aggregated, duplicated_labcodes = domain.aggregate_by_duplicates(
        full,
        prec=2,
        method=method,
    )
    assert duplicated_labcodes == []


def test_aggregate_by_duplicates_error():
    full = pd.DataFrame.from_dict(
        {
            "x1": [1.0, 2.0, 3.0, 1.0],
            "x2": [1.0, 2.0, 3.0, 1.0],
            "out1": [4.0, 5.0, 6.0, 3.0],
            "out2": [-4.0, -5.0, -6.0, -3.0],
            "valid_out1": [1, 1, 1, 1],
            "valid_out2": [1, 1, 1, 1],
        },
    )
    domain = Domain(
        inputs=Inputs(features=[if1, if2]),
        outputs=Outputs(features=[of1, of2]),
    )
    with pytest.raises(ValueError, match="Unknown aggregation type provided: 25"):
        domain.aggregate_by_duplicates(full, prec=2, method="25")


domain = Domain(
    inputs=Inputs(features=[if1, if2]),
    outputs=Outputs(features=[of1, of2, of1_, of2_]),
)


@pytest.mark.parametrize(
    "domain, feature_keys",
    [
        (domain, ["x1", "x2", "out1", "out2"]),
        (domain, ["x1", "x2", "out1"]),
        (domain, ["x1", "x2", "out1", "out3"]),
        (domain, ["x1", "out1", "out3", "out4"]),
        (domain2, ["x1", "x2", "out1", "out2"]),
    ],
)
def test_get_subdomain(domain, feature_keys):
    subdomain = get_subdomain(domain, feature_keys)
    assert (subdomain.inputs + subdomain.outputs).get_keys(Feature) == feature_keys


@pytest.mark.parametrize(
    "domain, output_feature_keys",
    [
        (domain, []),
        (domain, ["out1", "out2"]),
        (domain, ["x1", "x2"]),
        (domain, ["out1", "f1"]),
        (domain2, ["x1", "out1", "out2"]),
    ],
)
def test_get_subdomain_invalid(domain, output_feature_keys):
    with pytest.raises((AssertionError, ValueError, KeyError)):
        get_subdomain(domain, output_feature_keys)


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            pd.DataFrame(
                {
                    "col": [1, 2, 10, np.nan, "a"],
                    "col2": ["a", 10, 30, 40, 50],
                    "col3": [1, 2, 3, 4, 5.0],
                },
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "col": [1, 2, 10, np.nan, 6],
                    "col2": [5, 10, 30, 40, 50],
                    "col3": [1, 2, 3, 4, 5.0],
                },
            ),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "col": [1, 2, 10, 7.0, 6],
                    "col2": [5, 10, 30, 40, 50],
                    "col3": [1, 2, 3, 4, 5.0],
                },
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


def test_is_fulfilled():
    domain = Domain(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 0.8)),
            ContinuousInput(key="x2", bounds=(0, 1)),
        ],
        constraints=[
            LinearInequalityConstraint(
                features=["x1", "x2"],
                coefficients=[1.0, 1.0],
                rhs=1.0,
            ),
        ],
    )
    experiments = pd.DataFrame(
        {
            "x1": [0.5, 0.9, 0.6],
            "x2": [0.4, 0.05, 0.6],
        },
        index=[0, 2, 5],
    )
    assert_series_equal(
        domain.is_fulfilled(experiments),
        pd.Series([True, False, False], index=experiments.index),
    )
