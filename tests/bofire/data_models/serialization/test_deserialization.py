from pydantic import parse_obj_as

from bofire.data_models.api import (
    AnyAcquisitionFunction,
    AnyCondition,
    AnyConstraint,
    AnyFeature,
    AnyKernel,
    AnyMolFeatures,
    AnyObjective,
    AnyOutlierDetection,
    AnyPrior,
    AnyStrategy,
    AnySurrogate,
    Domain,
)
from tests.bofire.data_models.specs.api import Spec


def test_prior_should_be_deserializable(prior_spec: Spec):
    obj = prior_spec.obj()
    deserialized = parse_obj_as(AnyPrior, obj.dict())
    assert obj == deserialized


def test_kernel_should_be_deserializable(kernel_spec: Spec):
    obj = kernel_spec.obj()
    deserialized = parse_obj_as(AnyKernel, obj.dict())
    assert obj == deserialized


def test_constraint_should_be_deserializable(constraint_spec: Spec):
    obj = constraint_spec.obj()
    deserialized = parse_obj_as(AnyConstraint, obj.dict())
    assert obj == deserialized


def test_objective_should_be_deserializable(objective_spec: Spec):
    obj = objective_spec.obj()
    deserialized = parse_obj_as(AnyObjective, obj.dict())
    assert obj == deserialized


def test_feature_should_be_deserializable(feature_spec: Spec):
    obj = feature_spec.obj()
    deserialized = parse_obj_as(AnyFeature, obj.dict())
    assert obj == deserialized


def test_domain_should_be_deserializable(domain_spec: Spec):
    obj = domain_spec.obj()
    deserialized = parse_obj_as(Domain, obj.dict())
    assert obj == deserialized


def test_surrogate_should_be_deserializable(surrogate_spec: Spec):
    obj = surrogate_spec.obj()
    deserialized = parse_obj_as(AnySurrogate, obj.dict())
    assert obj == deserialized


def test_acquisition_function_should_be_deserializable(acquisition_function_spec: Spec):
    obj = acquisition_function_spec.obj()
    deserialized = parse_obj_as(AnyAcquisitionFunction, obj.dict())
    assert obj == deserialized


def test_strategy_should_be_deserializable(strategy_spec: Spec):
    obj = strategy_spec.obj()
    deserialized = parse_obj_as(AnyStrategy, obj.dict())
    # TODO: can we unhide the comparison of surrogate_specs?
    obj = {k: v for k, v in obj.dict().items() if k != "surrogate_specs"}
    deserialized = {
        k: v for k, v in deserialized.dict().items() if k != "surrogate_specs"
    }
    assert obj == deserialized


def test_condition_should_be_deserializable(condition_spec: Spec):
    obj = condition_spec.obj()
    deserialized = parse_obj_as(AnyCondition, obj.dict())
    assert obj == deserialized


def test_outlier_detection_should_be_deserializable(outlier_detection_spec: Spec):
    obj = outlier_detection_spec.obj()
    deserialized = parse_obj_as(AnyOutlierDetection, obj.dict())
    assert obj == deserialized


def test_molfeatures_should_be_deserializable(molfeatures_spec: Spec):
    obj = molfeatures_spec.obj()
    deserialized = parse_obj_as(AnyMolFeatures, obj.dict())
    assert obj == deserialized
