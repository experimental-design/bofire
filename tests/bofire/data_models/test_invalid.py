import pytest

from tests.bofire.data_models.specs.api import InvalidSpec


def _invalidate(invalid_spec: InvalidSpec):
    with pytest.raises(invalid_spec.error, match=invalid_spec.message):
        invalid_spec.cls(**invalid_spec.typed_spec())


def test_dataframe_should_be_invalid(invalid_dataframe_spec: InvalidSpec):
    _invalidate(invalid_dataframe_spec)


def test_feature_should_be_invalid(invalid_feature_spec: InvalidSpec):
    _invalidate(invalid_feature_spec)


def test_prior_should_be_invalid(invalid_prior_spec: InvalidSpec):
    _invalidate(invalid_prior_spec)


def test_prior_constraint_should_be_invalid(invalid_prior_constraint_spec: InvalidSpec):
    _invalidate(invalid_prior_constraint_spec)


def test_kernel_should_be_invalid(invalid_kernel_spec: InvalidSpec):
    _invalidate(invalid_kernel_spec)


def test_constraint_should_be_invalid(invalid_constraint_spec: InvalidSpec):
    _invalidate(invalid_constraint_spec)


def test_objective_should_be_invalid(invalid_objective_spec: InvalidSpec):
    _invalidate(invalid_objective_spec)


def test_domain_should_be_invalid(invalid_domain_spec: InvalidSpec):
    _invalidate(invalid_domain_spec)


def test_inputs_should_be_invalid(invalid_inputs_spec: InvalidSpec):
    _invalidate(invalid_inputs_spec)


def test_outputs_should_be_invalid(invalid_outputs_spec: InvalidSpec):
    _invalidate(invalid_outputs_spec)


def test_constraints_should_be_invalid(invalid_constraints_spec: InvalidSpec):
    _invalidate(invalid_constraints_spec)


def test_surrogate_should_be_invalid(invalid_surrogate_spec: InvalidSpec):
    _invalidate(invalid_surrogate_spec)


def test_acquisition_function_should_be_invalid(
    invalid_acquisition_function_spec: InvalidSpec,
):
    _invalidate(invalid_acquisition_function_spec)


def test_strategy_should_be_invalid(invalid_strategy_spec: InvalidSpec):
    _invalidate(invalid_strategy_spec)


def test_condition_should_be_invalid(invalid_condition_spec: InvalidSpec):
    _invalidate(invalid_condition_spec)


def test_transform_should_be_invalid(invalid_condition_spec: InvalidSpec):
    _invalidate(invalid_condition_spec)


def test_outlier_detection_should_be_invalid(
    invalid_outlier_detection_spec: InvalidSpec,
):
    _invalidate(invalid_outlier_detection_spec)


def test_molfeatures_should_be_invalid(invalid_molfeatures_spec: InvalidSpec):
    _invalidate(invalid_molfeatures_spec)


def test_local_search_config_should_be_invalid(
    invalid_local_search_config_spec: InvalidSpec,
):
    _invalidate(invalid_local_search_config_spec)
