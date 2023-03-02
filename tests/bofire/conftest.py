from pytest import fixture

from bofire.domain.constraint import Constraint
from bofire.domain.constraints import Constraints
from bofire.domain.domain import Domain
from bofire.domain.feature import Feature
from bofire.domain.features import Features
from bofire.domain.objective import Objective
from bofire.models.gps.kernels import BaseKernel
from bofire.models.gps.priors import Prior
from bofire.models.model import Model
from bofire.samplers import Sampler
from bofire.strategies.strategy import Strategy
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


# features


@fixture
def features() -> Features:
    return specs.featuress.valid().obj()


@fixture(params=specs.featuress.valids)
def valid_features_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.featuress.invalids)
def invalid_features_spec(request) -> specs.Spec:
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


# constraints


@fixture
def constraints() -> Constraints:
    return specs.constraintss.valid().obj()


@fixture(params=specs.constraintss.valids)
def valid_constraints_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.constraintss.invalids)
def invalid_constraints_spec(request) -> specs.Spec:
    return request.param


# model


@fixture
def model() -> Model:
    return specs.models.valid().obj()


@fixture(params=specs.models.valids)
def valid_model_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.models.invalids)
def invalid_model_spec(request) -> specs.Spec:
    return request.param


# domain


@fixture
def domain() -> Domain:
    return specs.domains.valid().obj()


@fixture(params=specs.domains.valids)
def valid_domain_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.domains.invalids)
def invalid_domain_spec(request) -> specs.Spec:
    return request.param


# prior


@fixture
def prior() -> Prior:
    return specs.priors.valid().obj()


@fixture(params=specs.priors.valids)
def valid_prior_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.priors.invalids)
def invalid_prior_spec(request) -> specs.Spec:
    return request.param


# kernel


@fixture
def kernel() -> BaseKernel:
    return specs.kernels.valid().obj()


@fixture(params=specs.kernels.valids)
def valid_kernel_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.kernels.invalids)
def invalid_kernel_spec(request) -> specs.Spec:
    return request.param


# sampler


@fixture
def sampler() -> Sampler:
    return specs.samplers.valid().obj()


@fixture(params=specs.samplers.valids)
def valid_sampler_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.samplers.invalids)
def invalid_sampler_spec(request) -> specs.Spec:
    return request.param


# strategy


@fixture
def strategy() -> Strategy:
    return specs.strategies.valid().obj()


@fixture(params=specs.strategies.valids)
def valid_strategy_spec(request) -> specs.Spec:
    return request.param


@fixture(params=specs.strategies.invalids)
def invalid_strategy_spec(request) -> specs.Spec:
    return request.param
