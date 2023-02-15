import json
from typing import List, Type

import pytest

from bofire.domain.constraints import Constraints
from bofire.domain.features import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    Features,
    InputFeatures,
    OutputFeatures,
)
from bofire.serial.serial import Deserialization
from tests.bofire import specs


def test_deserialization_should_process_constraint(valid_constraint_spec: specs.Spec):
    obj = valid_constraint_spec.obj()
    deserialized = Deserialization.constraint(valid_constraint_spec.typed_spec())
    assert isinstance(deserialized, valid_constraint_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_constraints():
    c1 = specs.constraints.valid().obj(key="c1")
    c2 = specs.constraints.valid().obj(key="c2")
    c3 = specs.constraints.valid().obj(key="c3")
    obj = Constraints(constraints=[c1, c2, c3])
    spec = {
        "type": "Constraints",
        "constraints": [
            json.loads(c1.json()),
            json.loads(c2.json()),
            json.loads(c3.json()),
        ],
    }
    deserialized = Deserialization.constraints(spec)
    assert isinstance(deserialized, Constraints)
    assert obj == deserialized


def test_deserialization_should_process_feature(valid_feature_spec: specs.Spec):
    obj = valid_feature_spec.obj()
    deserialized = Deserialization.feature(valid_feature_spec.typed_spec())
    assert isinstance(deserialized, valid_feature_spec.cls)
    assert obj == deserialized


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
def test_deserialization_should_process_features(
    cls: Type,
    feature_types: List[Type],
):
    features = [
        specs.features.valid(t).obj(key=f"f{i}") for i, t in enumerate(feature_types)
    ]
    obj = cls(features=features)
    spec = {
        "type": cls.__name__,
        "features": features,
    }
    deserialized = Deserialization.features(spec)
    assert isinstance(deserialized, Features)
    assert obj == deserialized


def test_deserialization_should_process_objective(valid_objective_spec: specs.Spec):
    obj = valid_objective_spec.obj()
    deserialized = Deserialization.objective(valid_objective_spec.typed_spec())
    assert isinstance(deserialized, valid_objective_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_model(valid_model_spec: specs.Spec):
    obj = valid_model_spec.obj()
    deserialized = Deserialization.model(valid_model_spec.typed_spec())
    assert isinstance(deserialized, valid_model_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_domain(valid_domain_spec: specs.Spec):
    obj = valid_domain_spec.obj()
    deserialized = Deserialization.domain(valid_domain_spec.typed_spec())
    assert isinstance(deserialized, valid_domain_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_kernel(valid_kernel_spec: specs.Spec):
    obj = valid_kernel_spec.obj()
    deserialized = Deserialization.kernel(valid_kernel_spec.typed_spec())
    assert isinstance(deserialized, valid_kernel_spec.cls)
    assert obj == deserialized
