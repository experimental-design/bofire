from pytest import fixture

import tests.bofire.data_models.specs.api as specs


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


try:
    # in case of the minimal installation these fixtures are not available
    @fixture(params=specs.strategies.valids)
    def strategy_spec(request) -> specs.Spec:
        return request.param

    @fixture(params=specs.surrogates.valids)
    def surrogate_spec(request) -> specs.Spec:
        return request.param

    @fixture(params=specs.priors.valids)
    def prior_spec(request) -> specs.Spec:
        return request.param

    @fixture(params=specs.kernels.valids)
    def kernel_spec(request) -> specs.Spec:
        return request.param

    @fixture(params=specs.conditions.valids)
    def condition_spec(request) -> specs.Spec:
        return request.param

    @fixture(params=specs.outlier_detection.valids)
    def outlier_detection_spec(request) -> specs.Spec:
        return request.param

    @fixture(params=specs.molfeatures.valids)
    def molfeatures_spec(request) -> specs.Spec:
        return request.param

except AttributeError:
    pass
