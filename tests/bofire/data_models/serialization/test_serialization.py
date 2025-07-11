from tests.bofire.data_models.specs.api import Spec


def test_dataframe_should_be_serializable(dataframe_spec: Spec):
    spec = dataframe_spec.typed_spec()
    obj = dataframe_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_prior_should_be_serializable(prior_spec: Spec):
    spec = prior_spec.typed_spec()
    obj = prior_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_prior_constraint_should_be_serializable(prior_constraint_spec: Spec):
    spec = prior_constraint_spec.typed_spec()
    obj = prior_constraint_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_kernel_should_be_serializable(kernel_spec: Spec):
    spec = kernel_spec.typed_spec()
    obj = kernel_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_constraint_should_be_serializable(constraint_spec: Spec):
    spec = constraint_spec.typed_spec()
    obj = constraint_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_objective_should_be_serializable(objective_spec: Spec):
    spec = objective_spec.typed_spec()
    obj = objective_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_feature_should_be_serializable(feature_spec: Spec):
    spec = feature_spec.typed_spec()
    obj = feature_spec.cls(**spec)
    assert obj.dict() == spec


def test_domain_should_be_serializable(domain_spec: Spec):
    spec = domain_spec.typed_spec()
    obj = domain_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_surrogate_should_be_serializable(surrogate_spec: Spec):
    spec = surrogate_spec.typed_spec()
    obj = surrogate_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_acquisition_function_should_be_serializable(acquisition_function_spec: Spec):
    spec = acquisition_function_spec.typed_spec()
    obj = acquisition_function_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_strategy_should_be_serializable(strategy_spec: Spec):
    spec = strategy_spec.typed_spec()
    obj = strategy_spec.cls(**spec)
    # TODO: can we unhide the comparison of surrogate_specs?
    data = {k: v for k, v in obj.model_dump().items() if k != "surrogate_specs"}
    for k, v in data.items():
        if v is not None:
            if hasattr(
                spec[k], "model_dump"
            ):  # works now for 1-time nested objects. Should be written recursively
                spec_k_dump = spec[k].model_dump()
                for kk, vv in v.items():
                    assert vv == spec_k_dump[kk]
            else:
                assert v == spec[k]


def test_condition_should_be_serializable(condition_spec: Spec):
    spec = condition_spec.typed_spec()
    obj = condition_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_outlier_detection_should_be_serializable(outlier_detection_spec: Spec):
    spec = outlier_detection_spec.typed_spec()
    obj = outlier_detection_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_molfeatures_should_be_serializable(molfeatures_spec: Spec):
    spec = molfeatures_spec.typed_spec()
    obj = molfeatures_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_inputs_should_be_serializable(inputs_spec: Spec):
    spec = inputs_spec.typed_spec()
    obj = inputs_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_outputs_should_be_serializable(outputs_spec: Spec):
    spec = outputs_spec.typed_spec()
    obj = outputs_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_constraints_should_be_serializable(constraints_spec: Spec):
    spec = constraints_spec.typed_spec()
    obj = constraints_spec.cls(**spec)
    assert obj.model_dump() == spec


def test_local_search_config_should_be_serializable(local_search_config_spec: Spec):
    spec = local_search_config_spec.typed_spec()
    obj = local_search_config_spec.cls(**spec)
    assert obj.model_dump() == spec
