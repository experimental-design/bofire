import itertools
from math import nan

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic.error_wrappers import ValidationError
from pytest import fixture

from bofire.data_models.api import Outputs
from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.domain.constraints import Constraints
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
    Input,
    Output,
)
from bofire.data_models.objectives.api import (
    ConstrainedObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    Objective,
    TargetObjective,
)
from bofire.utils.subdomain import get_subdomain


def test_empty_domain():
    assert Domain() == Domain(
        inputs=Inputs(), outputs=Outputs(), constraints=Constraints()
    )


class Bla(BaseModel):
    a: int


nf = Bla(a=1)

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
        )
    ]


def test_from_lists(input_list, output_list, constraint_list):
    assert Domain.from_lists(
        inputs=input_list, outputs=output_list, constraints=constraint_list
    ) == Domain(
        inputs=Inputs(features=input_list),
        outputs=Outputs(features=output_list),
        constraints=Constraints(constraints=constraint_list),
    )


@pytest.mark.parametrize(
    "inputs, outputs",
    [
        ([if1, if1_], []),
        ([], [of1, of1_]),
        ([if1], [of1]),
    ],
)
def test_duplicate_feature_names(inputs, outputs):
    with pytest.raises(ValidationError):
        Domain(
            inputs=inputs,
            outputs=outputs,
        )


@pytest.mark.parametrize(
    "inputs, outputs, constraints",
    [
        # input features
        ([nf], [], []),
        # output features
        ([], [nf], []),
        # constraints
        # ([], [], [nf]),
    ],
)
def test_invalid_domain_arg_types(inputs, outputs, constraints):
    with pytest.raises(Exception):
        Domain(
            inputs=inputs,
            outputs=outputs,
            constraints=constraints,
        )


@pytest.mark.parametrize(
    "inputs, constraints",
    [
        (
            [
                ContinuousInput(key="if1", bounds=(0, 1)),
                DiscreteInput(key="if2", values=[0.2, 0.7, 1.0]),
            ],
            [
                LinearEqualityConstraint(
                    features=["if1", "if2"], coefficients=[1.0, 1.0], rhs=11
                )
            ],
        )
    ],
)
def test_invalid_type_in_linear_constraints(inputs, constraints):
    with pytest.raises(ValidationError):
        Domain(inputs=inputs, constraints=constraints)


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
                )
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
                )
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
                )
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
                )
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
                )
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
                )
            ],
        ),
    ],
)
def test_unknown_features_in_domain(outputs, inputs, constraints):
    with pytest.raises(ValidationError):
        Domain(
            inputs=inputs,
            outputs=outputs,
            constraints=constraints,
        )


@pytest.mark.parametrize(
    "domain, data",
    [
        (
            Domain(),
            [],
        ),
        (
            Domain(
                inputs=Inputs(
                    features=[
                        CategoricalInput(
                            key="f1",
                            categories=["c11", "c12"],
                        ),
                    ]
                ),
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
            ],
        ),
        (
            Domain(
                inputs=Inputs(
                    features=[
                        CategoricalInput(
                            key="f1",
                            categories=["c11", "c12"],
                        ),
                        CategoricalInput(
                            key="f2",
                            categories=["c21", "c22", "c23"],
                        ),
                        CategoricalInput(
                            key="f3",
                            categories=["c31", "c32"],
                        ),
                    ]
                ),
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
                [("f2", "c21"), ("f2", "c22"), ("f2", "c23")],
                [("f3", "c31"), ("f3", "c32")],
            ],
        ),
    ],
)
def test_categorical_combinations_of_domain_defaults(domain, data):
    expected = list(itertools.product(*data))
    assert domain.inputs.get_categorical_combinations() == expected


@pytest.mark.parametrize(
    "domain, data, include, exclude",
    [
        (
            Domain(
                inputs=Inputs(
                    features=[
                        CategoricalInput(
                            key="f1",
                            categories=["c11", "c12"],
                        ),
                        CategoricalDescriptorInput(
                            key="f2",
                            categories=["c21", "c22"],
                            descriptors=["d21", "d22"],
                            values=[[1, 2], [3, 4]],
                        ),
                    ]
                ),
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
            ],
            CategoricalInput,
            CategoricalDescriptorInput,
        ),
        (
            Domain(
                inputs=Inputs(
                    features=[
                        CategoricalInput(
                            key="f1",
                            categories=["c11", "c12"],
                        ),
                        CategoricalDescriptorInput(
                            key="f2",
                            categories=["c21", "c22"],
                            descriptors=["d21", "d22"],
                            values=[[1, 2], [3, 4]],
                        ),
                    ]
                ),
            ),
            [
                [("f2", "c21"), ("f2", "c22")],
                [("f1", "c11"), ("f1", "c12")],
            ],
            CategoricalInput,
            None,
        ),
        (
            Domain(
                inputs=Inputs(
                    features=[
                        CategoricalInput(
                            key="f1",
                            categories=["c11", "c12"],
                        ),
                        CategoricalDescriptorInput(
                            key="f2",
                            categories=["c21", "c22"],
                            descriptors=["d21", "d22"],
                            values=[[1, 2], [3, 4]],
                        ),
                    ]
                ),
            ),
            [
                [("f2", "c21"), ("f2", "c22")],
            ],
            CategoricalDescriptorInput,
            None,
        ),
        (
            Domain(
                inputs=Inputs(
                    features=[
                        CategoricalInput(
                            key="f1",
                            categories=["c11", "c12"],
                        ),
                        CategoricalDescriptorInput(
                            key="f2",
                            categories=["c21", "c22"],
                            descriptors=["d21", "d22"],
                            values=[[1, 2], [3, 4]],
                        ),
                    ]
                ),
            ),
            [],
            CategoricalDescriptorInput,
            CategoricalInput,
        ),
    ],
)
def test_categorical_combinations_of_domain_filtered(domain, data, include, exclude):
    expected = list(itertools.product(*data))
    assert (
        domain.inputs.get_categorical_combinations(include=include, exclude=exclude)
        == expected
    )


data = pd.DataFrame.from_dict(
    {
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "out1": [nan, 1.0, 2.0, 3.0, nan, nan],
        "out2": [nan, 1.0, 2.0, 3.0, 4.0, 5.0],
        "valid_out1": [1, 0, 1, 1, 1, 1],
        "valid_out2": [1, 1, 0, 1, 1, 0],
    }
)

if1 = ContinuousInput(key="x1", bounds=(1, 10))
if2 = ContinuousInput(key="x2", bounds=(1, 10))

of1 = ContinuousOutput(key="out1", objective=obj)
of2 = ContinuousOutput(key="out2", objective=obj)

c1 = LinearEqualityConstraint(features=["x1", "x2"], coefficients=[5, 5], rhs=15.0)

of1_ = ContinuousOutput(key="out3", objective=None)
of2_ = ContinuousOutput(key="out4", objective=None)

domain = Domain(
    inputs=Inputs(features=[if1, if2]), outputs=Outputs(features=[of1, of2])
)
domain2 = Domain(
    inputs=Inputs(features=[if1, if2]),
    outputs=Outputs(features=[of1, of2]),
    constraints=Constraints(constraints=[c1]),
)


@pytest.mark.parametrize("domain", [domain, domain2])
def test_domain_serialize(domain):
    print("domain:", domain)
    import json

    print("dict:", json.dumps(domain.dict(), indent=4))
    ndomain = Domain(**json.loads(domain.json()))
    assert ndomain == domain


@pytest.mark.parametrize(
    "domain, data, output_feature_keys, expected",
    [
        (
            domain,
            data,
            None,
            pd.DataFrame.from_dict(
                {
                    "x1": [4.0],
                    "x2": [4.0],
                    "out1": [3.0],
                    "out2": [3.0],
                    "valid_out1": [1],
                    "valid_out2": [1],
                }
            ),
        ),
        (
            domain,
            data,
            [],
            pd.DataFrame.from_dict(
                {
                    "x1": [4.0],
                    "x2": [4.0],
                    "out1": [3.0],
                    "out2": [3.0],
                    "valid_out1": [1],
                    "valid_out2": [1],
                }
            ),
        ),
        (
            domain,
            data,
            ["out1", "out2"],
            pd.DataFrame.from_dict(
                {
                    "x1": [4.0],
                    "x2": [4.0],
                    "out1": [3.0],
                    "out2": [3.0],
                    "valid_out1": [1],
                    "valid_out2": [1],
                }
            ),
        ),
        (
            domain,
            data,
            ["out2"],
            pd.DataFrame.from_dict(
                {
                    "x1": [2.0, 4.0, 5.0],
                    "x2": [2.0, 4.0, 5.0],
                    "out1": [1.0, 3.0, nan],
                    "out2": [1.0, 3.0, 4.0],
                    "valid_out1": [0, 1, 1],
                    "valid_out2": [1, 1, 1],
                }
            ),
        ),
    ],
)
def test_preprocess_experiments_all_valid_outputs(
    domain, data, output_feature_keys, expected
):
    experiments = domain.outputs.preprocess_experiments_all_valid_outputs(
        data, output_feature_keys
    )
    assert_frame_equal(experiments.reset_index(drop=True), expected, check_dtype=False)


@pytest.mark.parametrize(
    "domain, data, expected",
    [
        (
            domain,
            data,
            pd.DataFrame.from_dict(
                {
                    "x1": [2, 3, 4, 5],
                    "x2": [2, 3, 4, 5],
                    "out1": [1, 2, 3, nan],
                    "out2": [1, 2, 3, 4],
                    "valid_out1": [0, 1, 1, 1],
                    "valid_out2": [1, 0, 1, 1],
                }
            ),
        )
    ],
)
def test_preprocess_experiments_any_valid_output(domain, data, expected):
    experiments = domain.outputs.preprocess_experiments_any_valid_output(data)
    assert experiments["x1"].tolist() == expected["x1"].tolist()
    assert experiments["out2"].tolist() == expected["out2"].tolist()


@pytest.mark.parametrize(
    "domain, data, expected",
    [
        (
            domain,
            data,
            pd.DataFrame.from_dict(
                {
                    "x1": [2, 4, 5],
                    "x2": [2, 4, 5],
                    "out1": [1, 3, nan],
                    "out2": [1, 3, 4],
                    "valid_out1": [0, 1, 1],
                    "valid_out2": [1, 1, 1],
                }
            ),
        )
    ],
)
def test_preprocess_experiments_one_valid_output(domain, data, expected):
    experiments = domain.outputs.preprocess_experiments_one_valid_output("out2", data)
    assert experiments["x1"].tolist() == expected["x1"].tolist()
    assert np.isnan(experiments["out1"].tolist()[2])
    assert experiments["out2"].tolist() == expected["out2"].tolist()


def test_coerce_invalids():
    domain = Domain(
        inputs=Inputs(features=[if1, if2]), outputs=Outputs(features=[of1, of2])
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
        }
    )
    assert_frame_equal(experiments, expected, check_dtype=False)


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
        }
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
        }
    )
    domain = Domain(
        inputs=Inputs(features=[if1, if2]), outputs=Outputs(features=[of1, of2])
    )
    aggregated, duplicated_labcodes = domain.aggregate_by_duplicates(
        full, prec=2, method=method
    )
    assert duplicated_labcodes == [["1", "4"]]
    assert_frame_equal(
        aggregated, expected_aggregated, check_dtype=False, check_like=True
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
        }
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
        }
    )
    domain = Domain(
        inputs=Inputs(features=[if1, if2]), outputs=Outputs(features=[of1, of2])
    )
    aggregated, duplicated_labcodes = domain.aggregate_by_duplicates(
        full, prec=2, method=method
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
        }
    )
    domain = Domain(
        inputs=Inputs(features=[if1, if2]), outputs=Outputs(features=[of1, of2])
    )
    with pytest.raises(ValueError, match="Unknown aggregation type provided: 25"):
        domain.aggregate_by_duplicates(full, prec=2, method="25")


domain = Domain(
    inputs=Inputs(features=[if1, if2]), outputs=Outputs(features=[of1, of2, of1_, of2_])
)


@pytest.mark.parametrize(
    "domain, FeatureType, exact, expected",
    [
        (domain, Output, True, []),
        (domain, Output, False, [of1, of2, of1_, of2_]),
        (domain, Output, None, [of1, of2, of1_, of2_]),
        (domain, ContinuousOutput, True, [of1, of2, of1_, of2_]),
        (domain, ContinuousOutput, False, [of1, of2, of1_, of2_]),
        (domain, ContinuousOutput, None, [of1, of2, of1_, of2_]),
        (domain, Input, True, []),
        (domain, Input, False, [if1, if2]),
        (domain, Input, None, [if1, if2]),
    ],
)
def test_get_features(domain, FeatureType, exact, expected):
    assert domain.get_features(FeatureType, exact=exact).features == expected


@pytest.mark.parametrize(
    "domain, includes, excludes, exact, expected",
    [
        (domain, [Objective], [], True, []),
        (domain, [Objective], [], False, [of1, of2]),
        (domain, [TargetObjective], [], False, [of1, of2]),
        (domain, [], [Objective], False, [of1_, of2_]),
    ],
)
def test_get_outputs_by_objective(domain: Domain, includes, excludes, exact, expected):
    assert (
        domain.outputs.get_by_objective(
            includes=includes,
            excludes=excludes,
            exact=exact,
        ).features
        == expected
    )


def test_get_outputs_by_objective_none():
    outputs = Outputs(
        features=[
            ContinuousOutput(key="a", objective=None),
            ContinuousOutput(
                key="b", objective=MaximizeSigmoidObjective(w=1, steepness=1, tp=0)
            ),
            ContinuousOutput(key="c", objective=MaximizeObjective()),
        ]
    )
    keys = outputs.get_keys_by_objective(excludes=ConstrainedObjective)
    assert keys == ["c"]
    assert outputs.get_keys().index("c") == 2
    assert outputs.get_keys_by_objective(excludes=Objective, includes=[]) == ["a"]
    assert outputs.get_by_objective(excludes=Objective, includes=[]) == Outputs(
        features=[ContinuousOutput(key="a", objective=None)]
    )


@pytest.mark.parametrize(
    "domain, FeatureType, exact, expected",
    [
        (domain, Output, True, []),
        (domain, Output, False, ["out1", "out2", "out3", "out4"]),
        (domain, Output, None, ["out1", "out2", "out3", "out4"]),
        (domain, ContinuousOutput, True, ["out1", "out2", "out3", "out4"]),
        (domain, ContinuousOutput, None, ["out1", "out2", "out3", "out4"]),
        (domain, Input, True, []),
        (domain, Input, False, ["x1", "x2"]),
        (domain, Input, None, ["x1", "x2"]),
    ],
)
def test_get_feature_keys(domain, FeatureType, exact, expected):
    assert domain.get_feature_keys(FeatureType, exact=exact) == expected


@pytest.mark.parametrize(
    "domain, includes, excludes, exact, expected",
    [
        (domain, [Objective], [], False, ["out1", "out2"]),
        (domain, [TargetObjective], [], False, ["out1", "out2"]),
        (domain, [Objective], [], True, []),
        (domain, [], [Objective], False, ["out3", "out4"]),
    ],
)
def test_get_output_keys_by_objective(
    domain: Domain, includes, excludes, exact, expected
):
    assert (
        domain.outputs.get_keys_by_objective(
            includes=includes,
            excludes=excludes,
            exact=exact,
        )
        == expected
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
    assert subdomain.get_feature_keys(Feature) == feature_keys


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
