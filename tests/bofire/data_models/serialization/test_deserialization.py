from pydantic import TypeAdapter

from bofire.data_models.api import (
    AnyAcquisitionFunction,
    AnyCondition,
    AnyConstraint,
    AnyDataFrame,
    AnyFeature,
    AnyKernel,
    AnyLocalSearchConfig,
    AnyMolFeatures,
    AnyObjective,
    AnyOutlierDetection,
    AnyPrior,
    AnyPriorConstraint,
    AnyStrategy,
    AnySurrogate,
    Constraints,
    Domain,
    Inputs,
    Outputs,
)
from tests.bofire.data_models.specs.api import Spec


def test_dataframe_should_be_deserializable(dataframe_spec: Spec):
    obj = dataframe_spec.obj()
    deserialized = TypeAdapter(AnyDataFrame).validate_python(obj.model_dump())
    assert obj == deserialized


def test_prior_should_be_deserializable(prior_spec: Spec):
    obj = prior_spec.obj()
    deserialized = TypeAdapter(AnyPrior).validate_python(obj.model_dump())
    assert obj == deserialized


def test_prior_constraint_should_be_deserializable(prior_constraint_spec: Spec):
    obj = prior_constraint_spec.obj()
    deserialized = TypeAdapter(AnyPriorConstraint).validate_python(obj.model_dump())
    assert obj == deserialized


def test_kernel_should_be_deserializable(kernel_spec: Spec):
    obj = kernel_spec.obj()
    deserialized = TypeAdapter(AnyKernel).validate_python(obj.model_dump())
    assert obj == deserialized


def test_constraint_should_be_deserializable(constraint_spec: Spec):
    obj = constraint_spec.obj()
    deserialized = TypeAdapter(AnyConstraint).validate_python(obj.model_dump())
    assert obj == deserialized


def test_objective_should_be_deserializable(objective_spec: Spec):
    obj = objective_spec.obj()
    deserialized = TypeAdapter(AnyObjective).validate_python(obj.model_dump())
    assert obj == deserialized


def test_feature_should_be_deserializable(feature_spec: Spec):
    obj = feature_spec.obj()
    deserialized = TypeAdapter(AnyFeature).validate_python(obj.model_dump())
    assert obj == deserialized


def test_domain_should_be_deserializable(domain_spec: Spec):
    obj = domain_spec.obj()
    deserialized = TypeAdapter(Domain).validate_python(obj.model_dump())
    assert obj == deserialized


def test_surrogate_should_be_deserializable(surrogate_spec: Spec):
    obj = surrogate_spec.obj()
    deserialized = TypeAdapter(AnySurrogate).validate_python(obj.model_dump())
    assert obj == deserialized


def test_acquisition_function_should_be_deserializable(acquisition_function_spec: Spec):
    obj = acquisition_function_spec.obj()
    deserialized = TypeAdapter(AnyAcquisitionFunction).validate_python(obj.model_dump())
    assert obj == deserialized


def test_strategy_should_be_deserializable(strategy_spec: Spec):
    obj = strategy_spec.obj()
    deserialized = TypeAdapter(AnyStrategy).validate_python(obj.model_dump())
    # TODO: can we unhide the comparison of surrogate_specs?
    obj = {k: v for k, v in obj.model_dump().items() if k != "surrogate_specs"}
    deserialized = {
        k: v for k, v in deserialized.model_dump().items() if k != "surrogate_specs"
    }
    assert obj == deserialized


def test_condition_should_be_deserializable(condition_spec: Spec):
    obj = condition_spec.obj()
    deserialized = TypeAdapter(AnyCondition).validate_python(obj.model_dump())
    assert obj == deserialized


def test_outlier_detection_should_be_deserializable(outlier_detection_spec: Spec):
    obj = outlier_detection_spec.obj()
    deserialized = TypeAdapter(AnyOutlierDetection).validate_python(obj.model_dump())
    assert obj == deserialized


def test_molfeatures_should_be_deserializable(molfeatures_spec: Spec):
    obj = molfeatures_spec.obj()
    deserialized = TypeAdapter(AnyMolFeatures).validate_python(obj.model_dump())
    assert obj == deserialized


def test_inputs_should_be_deserializable(inputs_spec: Spec):
    obj = inputs_spec.obj()
    deserialized = TypeAdapter(Inputs).validate_python(obj.model_dump())
    assert obj == deserialized


def test_outputs_should_be_deserializable(outputs_spec: Spec):
    obj = outputs_spec.obj()
    deserialized = TypeAdapter(Outputs).validate_python(obj.model_dump())
    assert obj == deserialized


def test_constraints_should_be_deserializable(constraints_spec: Spec):
    obj = constraints_spec.obj()
    deserialized = TypeAdapter(Constraints).validate_python(obj.model_dump())
    assert obj == deserialized


def test_local_search_config_should_be_deserializable(local_search_config_spec: Spec):
    obj = local_search_config_spec.obj()
    deserialized = TypeAdapter(AnyLocalSearchConfig).validate_python(obj.model_dump())
    assert obj == deserialized
