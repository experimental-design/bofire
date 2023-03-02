from bofire.serial.serialization import Serialization
from tests.bofire import specs


def test_serialization_should_jsonify_constraint(valid_constraint_spec: specs.Spec):
    obj = valid_constraint_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_constraint_spec.typed_spec()


def test_serialization_should_jsonify_constraints(valid_constraints_spec: specs.Spec):
    obj = valid_constraints_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_constraints_spec.typed_spec()


def test_serialization_should_jsonify_feature(valid_feature_spec: specs.Spec):
    obj = valid_feature_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_feature_spec.typed_spec()


def test_serialization_should_jsonify_features(valid_features_spec: specs.Spec):
    obj = valid_features_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_features_spec.typed_spec()


def test_serialization_should_jsonify_objective(valid_objective_spec: specs.Spec):
    obj = valid_objective_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_objective_spec.typed_spec()


def test_serialization_should_jsonify_model(valid_model_spec: specs.Spec):
    obj = valid_model_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_model_spec.typed_spec()


def test_serialization_should_jsonify_domain(valid_domain_spec: specs.Spec):
    print("spec:", valid_domain_spec)
    obj = valid_domain_spec.obj()
    print("obj:", obj)
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_domain_spec.typed_spec()


def test_serialization_should_jsonify_prior(valid_prior_spec: specs.Spec):
    obj = valid_prior_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_prior_spec.typed_spec()


def test_serialization_should_jsonify_kernel(valid_kernel_spec: specs.Spec):
    obj = valid_kernel_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_kernel_spec.typed_spec()


def test_serialization_should_jsonify_sampler(valid_sampler_spec: specs.Spec):
    obj = valid_sampler_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_sampler_spec.typed_spec()


def test_serialization_should_jsonify_strategy(valid_strategy_spec: specs.Spec):
    obj = valid_strategy_spec.obj()
    serialized = Serialization.json_dict(obj)
    assert serialized == valid_strategy_spec.typed_spec()
