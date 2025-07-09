from pytest import fixture

import tests.bofire.data_models.specs.api as specs


# invalid fixtures
@fixture(params=specs.dataframes.invalids)
def invalid_dataframe_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.local_search_configs.invalids)
def invalid_local_search_config_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.features.invalids)
def invalid_feature_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.priors.invalids)
def invalid_prior_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.prior_constraints.invalids)
def invalid_prior_constraint_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.kernels.invalids)
def invalid_kernel_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.constraints.invalids)
def invalid_constraint_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.objectives.invalids)
def invalid_objective_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.domain.invalids)
def invalid_domain_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.inputs.invalids)
def invalid_inputs_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.outputs.invalids)
def invalid_outputs_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.constraints_container.invalids)
def invalid_constraints_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.surrogates.invalids)
def invalid_surrogate_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.acquisition_functions.invalids)
def invalid_acquisition_function_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.strategies.invalids)
def invalid_strategy_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.conditions.invalids)
def invalid_condition_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.transforms.invalids)
def invalid_transforms_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.outlier_detection.invalids)
def invalid_outlier_detection_spec(request) -> specs.InvalidSpec:
    return request.param


@fixture(params=specs.molfeatures.invalids)
def invalid_molfeatures_spec(request) -> specs.InvalidSpec:
    return request.param


# valid fixtures
@fixture(params=specs.dataframes.valids)
def dataframe_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.constraints.valids)
def constraint_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.objectives.valids)
def objective_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.features.valids)
def feature_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.domain.valids)
def domain_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.acquisition_functions.valids)
def acquisition_function_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.inputs.valids)
def inputs_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.outputs.valids)
def outputs_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.constraints_container.valids)
def constraints_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.strategies.valids)
def strategy_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.surrogates.valids)
def surrogate_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.priors.valids)
def prior_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.prior_constraints.valids)
def prior_constraint_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.kernels.valids)
def kernel_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.conditions.valids)
def condition_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.transforms.valids)
def transforms_detection_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.outlier_detection.valids)
def outlier_detection_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.molfeatures.valids)
def molfeatures_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.local_search_configs.valids)
def local_search_config_spec(request) -> specs.Spec:
    return request.param
