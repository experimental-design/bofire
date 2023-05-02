from tests.bofire.data_models.specs.api import Spec


def test_prior_should_be_serializable(prior_spec: Spec):
    spec = prior_spec.typed_spec()
    obj = prior_spec.cls(**spec)
    assert obj.dict() == spec


def test_kernel_should_be_serializable(kernel_spec: Spec):
    spec = kernel_spec.typed_spec()
    obj = kernel_spec.cls(**spec)
    assert obj.dict() == spec


def test_constraint_should_be_serializable(constraint_spec: Spec):
    spec = constraint_spec.typed_spec()
    obj = constraint_spec.cls(**spec)
    assert obj.dict() == spec


def test_objective_should_be_serializable(objective_spec: Spec):
    spec = objective_spec.typed_spec()
    obj = objective_spec.cls(**spec)
    assert obj.dict() == spec


def test_feature_should_be_serializable(feature_spec: Spec):
    spec = feature_spec.typed_spec()
    obj = feature_spec.cls(**spec)
    print(spec)
    assert obj.dict() == spec


def test_domain_should_be_serializable(domain_spec: Spec):
    spec = domain_spec.typed_spec()
    obj = domain_spec.cls(**spec)
    assert obj.dict() == spec


def test_surrogate_should_be_serializable(surrogate_spec: Spec):
    spec = surrogate_spec.typed_spec()
    obj = surrogate_spec.cls(**spec)
    assert obj.dict() == spec


def test_acquisition_function_should_be_serializable(acquisition_function_spec: Spec):
    spec = acquisition_function_spec.typed_spec()
    obj = acquisition_function_spec.cls(**spec)
    assert obj.dict() == spec


def test_strategy_should_be_serializable(strategy_spec: Spec):
    spec = strategy_spec.typed_spec()
    obj = strategy_spec.cls(**spec)
    # TODO: can we unhide the comparison of surrogate_specs?
    data = {k: v for k, v in obj.dict().items() if k != "surrogate_specs"}
    for k, v in data.items():
        if v is not None:
            assert v == spec[k]
