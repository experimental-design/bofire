from bofire.serial.deserialization import Deserialization
from tests.bofire import specs


def test_deserialization_should_process_constraint(valid_constraint_spec: specs.Spec):
    obj = valid_constraint_spec.obj()
    deserialized = Deserialization.constraint(valid_constraint_spec.typed_spec())
    assert isinstance(deserialized, valid_constraint_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_constraints(valid_constraints_spec: specs.Spec):
    obj = valid_constraints_spec.obj()
    deserialized = Deserialization.constraints(valid_constraints_spec.typed_spec())
    assert isinstance(deserialized, valid_constraints_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_feature(valid_feature_spec: specs.Spec):
    obj = valid_feature_spec.obj()
    deserialized = Deserialization.feature(valid_feature_spec.typed_spec())
    assert isinstance(deserialized, valid_feature_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_features(valid_features_spec: specs.Spec):
    obj = valid_features_spec.obj()
    deserialized = Deserialization.features(valid_features_spec.typed_spec())
    assert isinstance(deserialized, valid_features_spec.cls)
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


def test_deserialization_should_process_prior(valid_prior_spec: specs.Spec):
    obj = valid_prior_spec.obj()
    deserialized = Deserialization.prior(valid_prior_spec.typed_spec())
    assert isinstance(deserialized, valid_prior_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_kernel(valid_kernel_spec: specs.Spec):
    obj = valid_kernel_spec.obj()
    deserialized = Deserialization.kernel(valid_kernel_spec.typed_spec())
    assert isinstance(deserialized, valid_kernel_spec.cls)
    assert obj == deserialized


def test_deserialization_should_process_sampler(valid_sampler_spec: specs.Spec):
    obj = valid_sampler_spec.obj()
    deserialized = Deserialization.sampler(valid_sampler_spec.typed_spec())
    assert isinstance(deserialized, valid_sampler_spec.cls)
    assert obj == deserialized
