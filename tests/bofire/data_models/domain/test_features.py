import pytest
from pydantic.error_wrappers import ValidationError

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.domain.api import Features, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
    Output,
)

# test features container
if1 = specs.features.valid(ContinuousInput).obj(key="if1")
if2 = specs.features.valid(ContinuousInput).obj(key="if2")
if3 = specs.features.valid(ContinuousInput).obj(key="if3", bounds=(3, 3))
if4 = specs.features.valid(CategoricalInput).obj(
    key="if4", categories=["a", "b"], allowed=[True, False]
)
if5 = specs.features.valid(DiscreteInput).obj(key="if5")
if7 = specs.features.valid(CategoricalInput).obj(
    key="if7",
    categories=["c", "d", "e"],
    allowed=[True, False, False],
)


of1 = specs.features.valid(ContinuousOutput).obj(key="of1")
of2 = specs.features.valid(ContinuousOutput).obj(key="of2")
of3 = specs.features.valid(ContinuousOutput).obj(key="of3", objective=None)

inputs = Inputs(features=[if1, if2])
inputs2 = Inputs(features=[if3, if4])
outputs = Outputs(features=[of1, of2])
outputs2 = Outputs(features=[of3])
features = Features(features=[if1, if2, of1, of2])
features2 = Features(features=[if3, if4, of3])


@pytest.mark.parametrize(
    "FeatureContainer, features",
    [
        (Features, ["s"]),
        (Features, [specs.features.valid(ContinuousInput).obj(), 5]),
        (Inputs, ["s"]),
        (Inputs, [specs.features.valid(ContinuousInput).obj(), 5]),
        (
            Inputs,
            [
                specs.features.valid(ContinuousInput).obj(),
                specs.features.valid(ContinuousOutput).obj(),
            ],
        ),
        (Outputs, ["s"]),
        (Outputs, [specs.features.valid(ContinuousOutput).obj(), 5]),
        (
            Outputs,
            [
                specs.features.valid(ContinuousOutput).obj(),
                specs.features.valid(ContinuousInput).obj(),
            ],
        ),
    ],
)
def test_features_invalid_feature(FeatureContainer, features):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        FeatureContainer(features=features)


@pytest.mark.parametrize(
    "features1, features2, expected_type",
    [
        [inputs, inputs2, Inputs],
        [outputs, outputs2, Outputs],
        [inputs, outputs, Features],
        [outputs, inputs, Features],
        [features2, outputs, Features],
        [features2, inputs, Features],
        [outputs, features2, Features],
        [inputs, features2, Features],
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
def test_features_get(features, FeatureType, exact, expected):
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
