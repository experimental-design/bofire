from pytest import fixture

from bofire.domain.constraints import Constraint
from bofire.domain.features import Feature
from bofire.domain.objectives import Objective
from tests.bofire import specs

# objective


@fixture
def objective() -> Objective:
    return specs.objectives.valid().obj()


@fixture(params=specs.objectives.valids)
def valid_objective_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.objectives.invalids)
def invalid_objective_spec(request) -> specs.Spec:
    return request.param


# feature


@fixture
def feature() -> Feature:
    return specs.features.valid().obj()


@fixture(params=specs.features.valids)
def valid_feature_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.features.invalids)
def invalid_feature_spec(request) -> specs.Spec:
    return request.param


# constraint


@fixture
def constraint() -> Constraint:
    return specs.constraints.valid().obj()


@fixture(params=specs.constraints.valids)
def valid_constraint_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.constraints.invalids)
def invalid_constraint_spec(request) -> specs.Spec:
    return request.param


# model


@fixture
def model() -> Constraint:
    return specs.models.valid().obj()


@fixture(params=specs.models.valids)
def valid_model_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.models.invalids)
def invalid_model_spec(request) -> specs.Spec:
    return request.param


# domain


@fixture
def domain() -> Constraint:
    return specs.domains.valid().obj()


@fixture(params=specs.domains.valids)
def valid_domain_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.domains.invalids)
def invalid_domain_spec(request) -> specs.Spec:
    return request.param


# prior


@fixture
def prior() -> Constraint:
    return specs.priors.valid().obj()


@fixture(params=specs.priors.valids)
def valid_prior_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.priors.invalids)
def invalid_prior_spec(request) -> specs.Spec:
    return request.param


# kernel


@fixture
def kernel() -> Constraint:
    return specs.kernels.valid().obj()


@fixture(params=specs.kernels.valids)
def valid_kernel_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.kernels.invalids)
def invalid_kernel_spec(request) -> specs.Spec:
    return request.param
