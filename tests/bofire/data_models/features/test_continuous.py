import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from pydantic.error_wrappers import ValidationError

import tests.bofire.data_models.specs.api as Specs
from bofire.data_models.domain.api import Features, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
    Feature,
    Output,
)


@pytest.mark.parametrize(
    "input_feature, expected, expected_value",
    [
        (Specs.features.valid(ContinuousInput).obj(key="k", bounds=(1, 1)), True, [1]),
        (
            Specs.features.valid(ContinuousInput).obj(key="k", bounds=(1, 2)),
            False,
            None,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(key="k", bounds=(2, 3)),
            False,
            None,
        ),
    ],
)
def test_continuous_input_feature_is_fixed(input_feature, expected, expected_value):
    assert input_feature.is_fixed() == expected
    assert input_feature.fixed_value() == expected_value


def test_continuous_input_invalid_stepsize():
    with pytest.raises(ValueError):
        Specs.features.valid(ContinuousInput).obj(key="a", bounds=(1, 1), stepsize=0)
    with pytest.raises(ValueError):
        Specs.features.valid(ContinuousInput).obj(key="a", bounds=(0, 5), stepsize=0.3)
    with pytest.raises(ValueError):
        Specs.features.valid(ContinuousInput).obj(key="a", bounds=(0, 1), stepsize=1)


def test_continuous_input_round():
    feature = Specs.features.valid(ContinuousInput).obj(key="a", bounds=(0, 5))
    values = pd.Series([1.0, 1.3, 0.55])
    assert_series_equal(values, feature.round(values))
    feature = Specs.features.valid(ContinuousInput).obj(
        key="a", bounds=(0, 5), stepsize=0.25
    )
    assert_series_equal(pd.Series([1.0, 1.25, 0.5]), feature.round(values))
    feature = Specs.features.valid(ContinuousInput).obj(
        key="a", bounds=(0, 5), stepsize=0.1
    )
    assert_series_equal(pd.Series([1.0, 1.3, 0.5]), feature.round(values))


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            Specs.features.valid(ContinuousInput).obj(key="if1", bounds=(0.5, 4)),
            (0.5, 4.0),
        ),
        (
            Specs.features.valid(ContinuousInput).obj(key="if1", bounds=(2.5, 2.9)),
            (1, 3.0),
        ),
        (Specs.features.valid(ContinuousInput).obj(key="if2", bounds=(1, 3)), (1, 3.0)),
        (Specs.features.valid(ContinuousInput).obj(key="if2", bounds=(1, 1)), (1, 1.0)),
    ],
)
def test_continuous_input_feature_get_bounds(input_feature, expected):
    experiments = pd.DataFrame({"if1": [1.0, 2.0, 3.0], "if2": [1.0, 1.0, 1.0]})
    lower, upper = input_feature.get_bounds(values=experiments[input_feature.key])
    assert (lower[0], upper[0]) == expected
    lower, upper = input_feature.get_bounds()
    assert (lower[0], upper[0]) == (
        input_feature.lower_bound,
        input_feature.upper_bound,
    )


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            Specs.features.valid(ContinuousInput).obj(),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            True,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            False,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            True,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
            False,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([3.0, 3.0, 3.0]),
            False,
        ),
    ],
)
def test_continuous_input_feature_validate_valid(input_feature, values, strict):
    input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            Specs.features.valid(ContinuousInput).obj(),
            pd.Series([3.0, "mama"]),
            True,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(),
            pd.Series([3.0, "mama"]),
            False,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([3.0, 3.0, 3.0]),
            True,
        ),
    ],
)
def test_continuous_input_feature_validate_invalid(input_feature, values, strict):
    with pytest.raises(ValueError):
        input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            Specs.features.valid(ContinuousInput).obj(),
            pd.Series([random.uniform(3.0, 5.3) for _ in range(20)]),
        ),
        (
            Specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([3.0, 3.0, 3.0]),
        ),
    ],
)
def test_continuous_input_feature_validate_candidental_valid(input_feature, values):
    input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            Specs.features.valid(ContinuousInput).obj(),
            pd.Series([3.1, "a"]),
        ),
        (
            Specs.features.valid(ContinuousInput).obj(),
            pd.Series([2.9, 4.0]),
        ),
        (
            Specs.features.valid(ContinuousInput).obj(),
            pd.Series([4.0, 6]),
        ),
        (
            Specs.features.valid(ContinuousInput).obj(bounds=(3, 3)),
            pd.Series([3.1, 3.2, 3.4]),
        ),
    ],
)
def test_continuous_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "feature, xt, expected",
    [
        (
            Specs.features.valid(ContinuousInput).obj(key="a", bounds=(0, 10)),
            pd.Series(np.linspace(0, 1, 11)),
            np.linspace(0, 10, 11),
        ),
        (
            Specs.features.valid(ContinuousInput).obj(key="a", bounds=(-10, 20)),
            pd.Series(np.linspace(0, 1)),
            np.linspace(-10, 20),
        ),
    ],
)
def test_continuous_input_feature_from_unit_range(feature, xt, expected):
    x = feature.from_unit_range(xt)
    assert np.allclose(x.values, expected)


@pytest.mark.parametrize(
    "feature, x, expected, real",
    [
        (
            Specs.features.valid(ContinuousInput).obj(key="a", bounds=(0, 10)),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            True,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(key="a", bounds=(-10, 20)),
            pd.Series(np.linspace(-10, 20)),
            np.linspace(0, 1),
            True,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(key="a", bounds=(0, 10)),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            False,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(key="a", bounds=(-10, 20)),
            pd.Series(np.linspace(-10, 20)),
            np.linspace(0, 1),
            False,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(key="a", bounds=(0, 9)),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 1, 11),
            True,
        ),
        (
            Specs.features.valid(ContinuousInput).obj(key="a", bounds=(0, 9)),
            pd.Series(np.linspace(0, 10, 11)),
            np.linspace(0, 10 / 9, 11),
            False,
        ),
    ],
)
def test_continuous_input_feature_to_unit_range(feature, x, expected, real):
    xt = feature.to_unit_range(x)
    assert np.allclose(xt.values, expected, real)


@pytest.mark.parametrize(
    "FeatureContainer, features",
    [
        (Features, ["s"]),
        (Features, [Specs.features.valid(ContinuousInput).obj(), 5]),
        (Inputs, ["s"]),
        (Inputs, [Specs.features.valid(ContinuousInput).obj(), 5]),
        (
            Inputs,
            [
                Specs.features.valid(ContinuousInput).obj(),
                Specs.features.valid(ContinuousOutput).obj(),
            ],
        ),
        (Outputs, ["s"]),
        (Outputs, [Specs.features.valid(ContinuousOutput).obj(), 5]),
        (
            Outputs,
            [
                Specs.features.valid(ContinuousOutput).obj(),
                Specs.features.valid(ContinuousInput).obj(),
            ],
        ),
    ],
)
def test_features_invalid_feature(FeatureContainer, features):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        FeatureContainer(features=features)


# test features container
if1 = Specs.features.valid(ContinuousInput).obj(key="if1")
if2 = Specs.features.valid(ContinuousInput).obj(key="if2")
if3 = Specs.features.valid(ContinuousInput).obj(key="if3", bounds=(3, 3))

of1 = Specs.features.valid(ContinuousOutput).obj(key="of1")
of2 = Specs.features.valid(ContinuousOutput).obj(key="of2")
of3 = Specs.features.valid(ContinuousOutput).obj(key="of3", objective=None)

inputs = Inputs(features=[if1, if2])
outputs = Outputs(features=[of1, of2])
features = Features(features=[if1, if2, of1, of2])


@pytest.mark.parametrize(
    "features1, features2, expected_type",
    [
        [inputs, inputs, Inputs],
        [outputs, outputs, Outputs],
        [inputs, outputs, Features],
        [outputs, inputs, Features],
        [features, outputs, Features],
        [features, inputs, Features],
        [outputs, features, Features],
        [inputs, features, Features],
    ],
)
def test_features_plus(features1, features2, expected_type):
    returned = features1 + features2
    assert type(returned) == expected_type
    assert len(returned) == (len(features1) + len(features2))


@pytest.mark.parametrize(
    "features, FeatureType, exact, expected",
    [
        (features, Feature, False, [if1, if2, of1, of2]),
        (features, Output, False, [of1, of2]),
        (inputs, ContinuousInput, False, [if1, if2]),
        (outputs, ContinuousOutput, False, [of1, of2]),
    ],
)
def test_constraints_get(features, FeatureType, exact, expected):
    returned = features.get(FeatureType, exact=exact)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])
    assert type(returned) == type(features)


@pytest.mark.parametrize(
    "features, FeatureType, exact, expected",
    [
        (features, Feature, False, ["if1", "if2", "of1", "of2"]),
        (features, Output, False, ["of1", "of2"]),
        (inputs, ContinuousInput, False, ["if1", "if2"]),
        (outputs, ContinuousOutput, False, ["of1", "of2"]),
    ],
)
def test_features_get_keys(features, FeatureType, exact, expected):
    assert features.get_keys(FeatureType, exact=exact) == expected


@pytest.mark.parametrize(
    "features, key, expected",
    [
        (features, "if1", if1),
        (outputs, "of1", of1),
        (inputs, "if1", if1),
    ],
)
def test_features_get_by_key(features, key, expected):
    returned = features.get_by_key(key)
    assert returned.key == expected.key
    assert id(returned) == id(expected)


def test_features_get_by_keys():
    keys = ["of2", "if1"]
    feats = features.get_by_keys(keys)
    assert feats[0].key == "if1"
    assert feats[1].key == "of2"


@pytest.mark.parametrize(
    "features, key",
    [
        (features, "if133"),
        (outputs, "of3331"),
        (inputs, "if1333333"),
    ],
)
def test_features_get_by_key_invalid(features, key):
    with pytest.raises(KeyError):
        features.get_by_key(key)
