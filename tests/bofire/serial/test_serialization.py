import json
from typing import List, Type

import pytest

from bofire.domain.constraints import Constraints
from bofire.domain.features import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    InputFeatures,
    OutputFeatures,
)
from bofire.serial.serial import Serialization
from tests.bofire import specs


def test_serialization_should_jsonify_constraint(valid_constraint_spec: specs.Spec):
    obj = valid_constraint_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_constraint_spec.typed_spec()


def test_serialization_should_jsonify_constraints():
    c1 = specs.constraints.valid().obj(key="c1")
    c2 = specs.constraints.valid().obj(key="c2")
    c3 = specs.constraints.valid().obj(key="c3")
    obj = Constraints(constraints=[c1, c2, c3])
    serialized = Serialization.json_dict(obj)
    assert serialized["constraints"] == [
        json.loads(c1.json()),
        json.loads(c2.json()),
        json.loads(c3.json()),
    ]


def test_serialization_should_jsonify_feature(valid_feature_spec: specs.Spec):
    obj = valid_feature_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_feature_spec.typed_spec()


@pytest.mark.parametrize(
    "cls, feature_types",
    [
        (
            InputFeatures,
            [ContinuousInput, CategoricalInput, ContinuousInput, CategoricalInput],
        ),
        (OutputFeatures, [ContinuousOutput, ContinuousOutput]),
    ],
)
def test_serialization_should_jsonify_features(
    cls: Type,
    feature_types: List[Type],
):
    features = [
        specs.features.valid(t).obj(key=f"f{i}") for i, t in enumerate(feature_types)
    ]
    obj = cls(features=features)
    serialized = Serialization.json_dict(obj)
    assert serialized["features"] == [json.loads(f.json()) for f in features]


def test_serialization_should_jsonify_objective(valid_objective_spec: specs.Spec):
    obj = valid_objective_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_objective_spec.typed_spec()


def test_serialization_should_jsonify_model(valid_model_spec: specs.Spec):
    obj = valid_model_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_model_spec.typed_spec()


def test_serialization_should_jsonify_domain(valid_domain_spec: specs.Spec):
    obj = valid_domain_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_domain_spec.typed_spec()
