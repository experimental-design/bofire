import itertools
from math import nan

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic.error_wrappers import ValidationError

from bofire.domain.constraints import LinearEqualityConstraint, NChooseKConstraint
from bofire.domain.desirability_functions import (
    DesirabilityFunction,
    TargetDesirabilityFunction,
)
from bofire.domain.domain import Domain, get_subdomain
from bofire.domain.features import (
    CategoricalDescriptorInputFeature,
    CategoricalInputFeature,
    ContinuousInputFeature,
    ContinuousOutputFeature,
    ContinuousOutputFeature_woDesFunc,
    Feature,
    InputFeature,
    OutputFeature,
)
from bofire.domain.util import BaseModel


def test_empty_domain():
    Domain(
        input_features=[],
        output_features=[],
        constraints=[],
    )


df = TargetDesirabilityFunction(
    target_value=1,
    steepness=2,
    tolerance=3,
    w=0.5,
)


class Bla(BaseModel):
    a: int


nf = Bla(a=1)

if_ = CategoricalInputFeature(key="if", categories=["a", "b"])
of_ = ContinuousOutputFeature(key="of", desirability_function=df)


if1 = ContinuousInputFeature(key="f1", upper_bound=10, lower_bound=0)
if1_ = ContinuousInputFeature(key="f1", upper_bound=10, lower_bound=1)
if2 = ContinuousInputFeature(key="f2", upper_bound=10, lower_bound=0)

of1 = ContinuousOutputFeature(key="f1", desirability_function=df)
of1_ = ContinuousOutputFeature(key="f1", desirability_function=df)
of2 = ContinuousOutputFeature(key="f2", desirability_function=df)

of3 = ContinuousOutputFeature(key="of3", desirability_function=df)


@pytest.mark.parametrize(
    "input_features, output_features",
    [
        ([if1, if1_], []),
        ([], [of1, of1_]),
        ([if1], [of1]),
    ],
)
def test_duplicate_feature_names(input_features, output_features):
    with pytest.raises(ValidationError):
        Domain(
            input_features=input_features,
            output_features=output_features,
            constraints=[],
        )


@pytest.mark.parametrize(
    "input_features, output_features, constraints",
    [
        # input features
        ([nf], [], []),
        # output features
        ([], [nf], []),
        # constraints
        # ([], [], [nf]),
    ],
)
def test_invalid_domain_arg_types(input_features, output_features, constraints):
    with pytest.raises(Exception):
        Domain(
            input_features=input_features,
            output_features=output_features,
            constraints=constraints,
        )


@pytest.mark.parametrize(
    "input_features, output_features, constraints",
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
def test_valid_constraints_in_domain(output_features, input_features, constraints):
    Domain(
        input_features=input_features,
        output_features=output_features,
        constraints=constraints,
    )


@pytest.mark.parametrize(
    "input_features, output_features, constraints",
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
def test_unknown_features_in_domain(output_features, input_features, constraints):
    with pytest.raises(ValidationError):
        Domain(
            input_features=input_features,
            output_features=output_features,
            constraints=constraints,
        )


@pytest.mark.parametrize(
    "domain, data",
    [
        (
            Domain(
                input_features=[],
                output_features=[],
                constraints=[],
            ),
            [],
        ),
        (
            Domain(
                input_features=[
                    CategoricalInputFeature(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                ],
                output_features=[],
                constraints=[],
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
            ],
        ),
        (
            Domain(
                input_features=[
                    CategoricalInputFeature(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalInputFeature(
                        key="f2",
                        categories=["c21", "c22", "c23"],
                    ),
                    CategoricalInputFeature(
                        key="f3",
                        categories=["c31", "c32"],
                    ),
                ],
                output_features=[],
                constraints=[],
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
    assert domain.get_categorical_combinations() == expected


@pytest.mark.parametrize(
    "domain, data, include, exclude",
    [
        (
            Domain(
                input_features=[
                    CategoricalInputFeature(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalDescriptorInputFeature(
                        key="f2",
                        categories=["c21", "c22"],
                        descriptors=["d21", "d22"],
                        values=[[1, 2], [3, 4]],
                    ),
                ],
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
            ],
            CategoricalInputFeature,
            CategoricalDescriptorInputFeature,
        ),
        (
            Domain(
                input_features=[
                    CategoricalInputFeature(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalDescriptorInputFeature(
                        key="f2",
                        categories=["c21", "c22"],
                        descriptors=["d21", "d22"],
                        values=[[1, 2], [3, 4]],
                    ),
                ],
            ),
            [
                [("f1", "c11"), ("f1", "c12")],
                [("f2", "c21"), ("f2", "c22")],
            ],
            CategoricalInputFeature,
            None,
        ),
        (
            Domain(
                input_features=[
                    CategoricalInputFeature(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalDescriptorInputFeature(
                        key="f2",
                        categories=["c21", "c22"],
                        descriptors=["d21", "d22"],
                        values=[[1, 2], [3, 4]],
                    ),
                ],
            ),
            [
                [("f2", "c21"), ("f2", "c22")],
            ],
            CategoricalDescriptorInputFeature,
            None,
        ),
        (
            Domain(
                input_features=[
                    CategoricalInputFeature(
                        key="f1",
                        categories=["c11", "c12"],
                    ),
                    CategoricalDescriptorInputFeature(
                        key="f2",
                        categories=["c21", "c22"],
                        descriptors=["d21", "d22"],
                        values=[[1, 2], [3, 4]],
                    ),
                ],
            ),
            [],
            CategoricalDescriptorInputFeature,
            CategoricalInputFeature,
        ),
    ],
)
def test_categorical_combinations_of_domain_filtered(domain, data, include, exclude):
    expected = list(itertools.product(*data))
    assert (
        domain.get_categorical_combinations(include=include, exclude=exclude)
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

if1 = ContinuousInputFeature(key="x1", upper_bound=10, lower_bound=1)
if2 = ContinuousInputFeature(key="x2", upper_bound=10, lower_bound=1)

of1 = ContinuousOutputFeature(key="out1", desirability_function=df)
of2 = ContinuousOutputFeature(key="out2", desirability_function=df)

c1 = LinearEqualityConstraint(features=["x1", "x2"], coefficients=[5, 5], rhs=15.0)

of1_ = ContinuousOutputFeature_woDesFunc(key="out3")
of2_ = ContinuousOutputFeature_woDesFunc(key="out4")

domain = Domain(input_features=[if1, if2], output_features=[of1, of2])
domain2 = Domain(
    input_features=[if1, if2], output_features=[of1, of2], constraints=[c1]
)


@pytest.mark.parametrize("domain", [domain, domain2])
def test_domain_serialie(domain):
    config = domain.to_config()
    ndomain = Domain.from_config(config=config)
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
    experiments = domain.preprocess_experiments_all_valid_outputs(
        data, output_feature_keys
    )
    assert_frame_equal(experiments.reset_index(drop=True), expected, check_dtype=False)


def test_preprocess_experiments_all_valid_outputs_invalid():
    with pytest.raises(AssertionError):
        _ = domain.preprocess_experiments_all_valid_outputs(
            data, output_feature_keys=["x1"]
        )


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
    experiments = domain.preprocess_experiments_any_valid_output(data)
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
    experiments = domain.preprocess_experiments_one_valid_output("out2", data)
    assert experiments["x1"].tolist() == expected["x1"].tolist()
    assert np.isnan(experiments["out1"].tolist()[2])
    assert experiments["out2"].tolist() == expected["out2"].tolist()


def test_coerce_invalids():
    domain = Domain(input_features=[if1, if2], output_features=[of1, of2])
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


def test_aggregate_by_duplicates():
    # dataframe with duplicates
    full = pd.DataFrame.from_dict(
        {
            "x1": [1.0, 2.0, 3.0, 1.0],
            "x2": [1.0, 2.0, 3.0, 1.0],
            "out1": [
                4.0,
                5.0,
                6.0,
                3.0,
            ],
            "out2": [
                -4.0,
                -5.0,
                -6.0,
                -3.0,
            ],
            "valid_out1": [1, 1, 1, 1],
            "valid_out2": [1, 1, 1, 1],
        }
    )
    expected_aggregated = pd.DataFrame.from_dict(
        {
            "labcode": ["1-4", "2", "3"],
            "x1": [1.0, 2.0, 3.0],
            "x2": [
                1.0,
                2.0,
                3.0,
            ],
            "out1": [3.5, 5.0, 6.0],
            "out2": [-3.5, -5.0, -6.0],
            "valid_out1": [1, 1, 1],
            "valid_out2": [1, 1, 1],
        }
    )
    domain = Domain(input_features=[if1, if2], output_features=[of1, of2])
    aggregated, duplicated_labcodes = domain.aggregate_by_duplicates(full, prec=2)
    assert duplicated_labcodes == [["1", "4"]]
    assert_frame_equal(
        aggregated, expected_aggregated, check_dtype=False, check_like=True
    )
    # dataset without duplicates
    full = pd.DataFrame.from_dict(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [1.0, 2.0, 3.0, 4.0],
            "out1": [
                4.0,
                5.0,
                6.0,
                3.0,
            ],
            "out2": [
                -4.0,
                -5.0,
                -6.0,
                -3.0,
            ],
            "valid_out1": [1, 1, 1, 1],
            "valid_out2": [1, 1, 1, 1],
        }
    )
    expected_aggregated = pd.DataFrame.from_dict(
        {
            "labcode": ["0", "1", "2", "3"],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [1.0, 2.0, 3.0, 4.0],
            "out1": [
                4.0,
                5.0,
                6.0,
                3.0,
            ],
            "out2": [
                -4.0,
                -5.0,
                -6.0,
                -3.0,
            ],
            "valid_out1": [1, 1, 1, 1],
            "valid_out2": [1, 1, 1, 1],
        }
    )
    domain = Domain(input_features=[if1, if2], output_features=[of1, of2])
    aggregated, duplicated_labcodes = domain.aggregate_by_duplicates(full, prec=2)
    assert duplicated_labcodes == []


domain = Domain(input_features=[if1, if2], output_features=[of1, of2, of1_, of2_])


@pytest.mark.parametrize(
    "domain, FeatureType, exact, expected",
    [
        (domain, OutputFeature, True, []),
        (domain, OutputFeature, False, [of1, of2, of1_, of2_]),
        (domain, OutputFeature, None, [of1, of2, of1_, of2_]),
        (domain, ContinuousOutputFeature, True, [of1, of2]),
        (domain, ContinuousOutputFeature, False, [of1, of2, of1_, of2_]),
        (domain, ContinuousOutputFeature, None, [of1, of2, of1_, of2_]),
        (domain, ContinuousOutputFeature_woDesFunc, True, [of1_, of2_]),
        (domain, ContinuousOutputFeature_woDesFunc, False, [of1_, of2_]),
        (domain, ContinuousOutputFeature_woDesFunc, None, [of1_, of2_]),
        (domain, InputFeature, True, []),
        (domain, InputFeature, False, [if1, if2]),
        (domain, InputFeature, None, [if1, if2]),
    ],
)
def test_get_features(domain, FeatureType, exact, expected):
    assert domain.get_features(FeatureType, exact=exact) == expected


@pytest.mark.parametrize(
    "domain, DesirabilityType, exact, expected",
    [
        (domain, DesirabilityFunction, True, []),
        (domain, DesirabilityFunction, False, [of1, of2, of1_, of2_]),
        (domain, TargetDesirabilityFunction, False, [of1, of2]),
    ],
)
def test_get_by_desirability(
    domain: Domain, DesirabilityType: DesirabilityFunction, exact, expected
):
    assert (
        domain.get_outputs_by_desirability(
            DesirabilityType,
            exact=exact,
        )
        == expected
    )


@pytest.mark.parametrize(
    "domain, FeatureType, exact, expected",
    [
        (domain, OutputFeature, True, []),
        (domain, OutputFeature, False, ["out1", "out2", "out3", "out4"]),
        (domain, OutputFeature, None, ["out1", "out2", "out3", "out4"]),
        (domain, ContinuousOutputFeature, True, ["out1", "out2"]),
        (domain, ContinuousOutputFeature, None, ["out1", "out2", "out3", "out4"]),
        (domain, ContinuousOutputFeature_woDesFunc, True, ["out3", "out4"]),
        (domain, ContinuousOutputFeature_woDesFunc, False, ["out3", "out4"]),
        (domain, ContinuousOutputFeature_woDesFunc, None, ["out3", "out4"]),
        (domain, InputFeature, True, []),
        (domain, InputFeature, False, ["x1", "x2"]),
        (domain, InputFeature, None, ["x1", "x2"]),
    ],
)
def test_get_feature_keys(domain, FeatureType, exact, expected):
    assert domain.get_feature_keys(FeatureType, exact=exact) == expected


@pytest.mark.parametrize(
    "domain, DesirabilityType, exact, expected",
    [
        (domain, DesirabilityFunction, False, ["out1", "out2", "out3", "out4"]),
        (domain, TargetDesirabilityFunction, False, ["out1", "out2"]),
        (domain, DesirabilityFunction, True, []),
    ],
)
def test_get_key_by_desirability(
    domain: Domain, DesirabilityType: DesirabilityFunction, exact, expected
):
    assert (
        domain.get_output_keys_by_desirability(
            DesirabilityType,
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
