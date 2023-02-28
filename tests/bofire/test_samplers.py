import pytest

from bofire.domain.constraint import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain.constraints import Constraints
from bofire.domain.domain import Domain
from bofire.domain.feature import CategoricalInput, ContinuousInput
from bofire.domain.features import InputFeatures
from bofire.samplers import PolytopeSampler, RejectionSampler

input_features = InputFeatures(
    features=[
        ContinuousInput(key=f"if{i}", lower_bound=0, upper_bound=1) for i in range(1, 4)
    ]
)
constraints = Constraints(
    constraints=[
        LinearInequalityConstraint(
            features=["if1", "if2", "if3"], coefficients=[1, 1, 1], rhs=1
        )
    ]
)


@pytest.mark.parametrize(
    "features, constraints, sampling_method, num_samples",
    [
        (input_features, constraints, sampling_method, num_samples)
        for sampling_method in ["SOBOL", "UNIFORM", "LHS"]
        for num_samples in [1, 2, 64, 128]
    ],
)
def test_rejection_sampler(features, constraints, sampling_method, num_samples):
    domain = Domain(
        input_features=features,
        constraints=constraints,
    )
    sampler = RejectionSampler(domain=domain, sampling_method=sampling_method)
    sampler.ask(num_samples)


def test_rejection_sampler_not_converged():
    domain = Domain(
        input_features=input_features,
        constraints=constraints,
    )
    sampler = RejectionSampler(domain=domain, num_base_samples=16, max_iters=2)
    with pytest.raises(ValueError):
        sampler.ask(128)


if1 = ContinuousInput(
    lower_bound=0.0,
    upper_bound=1.0,
    key="if1",
)
if2 = ContinuousInput(
    lower_bound=0.0,
    upper_bound=1.0,
    key="if2",
)
if3 = ContinuousInput(
    lower_bound=0.0,
    upper_bound=1.0,
    key="if3",
)
if4 = ContinuousInput(
    lower_bound=0.1,
    upper_bound=0.1,
    key="if4",
)
if5 = CategoricalInput(
    categories=["a", "b", "c"],
    key="if5",
)
if6 = CategoricalInput(
    categories=["a", "b", "c"],
    allowed=[False, True, False],
    key="if6",
)
If7 = ContinuousInput(lower_bound=1.0, upper_bound=1.0, key="If7")
c1 = LinearEqualityConstraint(
    features=["if1", "if2", "if3", "if4"], coefficients=[1.0, 1.0, 1.0, 1.0], rhs=1.0
)
c2 = LinearInequalityConstraint.from_greater_equal(
    features=["if1", "if2"], coefficients=[1.0, 1.0], rhs=0.2
)
c3 = LinearInequalityConstraint.from_greater_equal(
    features=["if1", "if2", "if4"], coefficients=[1.0, 1.0, 0.5], rhs=0.2
)
c4 = LinearEqualityConstraint(
    features=["if1", "if2", "if3", "if4", "If7"],
    coefficients=[1.0, 1.0, 1.0, 1.0, 1.0],
    rhs=2.0,
)
c5 = LinearEqualityConstraint(features=["if1", "if4"], coefficients=[1.0, 1.0], rhs=1.0)
c6 = NChooseKConstraint(
    features=["if1", "if2", "if3"],
    min_count=1,
    max_count=2,
    none_also_valid=False,
)

domains = [
    Domain(input_features=[if1, if2, if3], constraints=[c2]),
    Domain(input_features=[if1, if2, if3, if4], constraints=[c1, c2]),
    Domain(input_features=[if1, if2, if3, if4], constraints=[c1, c2, c3]),
    Domain(input_features=[if1, if2, if3, if5], constraints=[c2]),
    Domain(input_features=[if1, if2, if3, if4, if5], constraints=[c1, c2]),
    Domain(input_features=[if1, if2, if3, if4, if5], constraints=[c1, c2, c3]),
    Domain(input_features=[if1, if2, if3, if6], constraints=[c2]),
    Domain(input_features=[if1, if2, if3, if4, if6], constraints=[c1, c2]),
    Domain(input_features=[if1, if2, if3, if4, if6], constraints=[c1, c2, c3]),
    Domain(input_features=[if1, if2, if3, if4, if6, If7], constraints=[c1, c2, c3]),
    Domain(input_features=[if1, if2, if3, if4, if6, If7], constraints=[c1, c2, c3, c4]),
    Domain(input_features=[if1, if2, if3, if4], constraints=[c5]),
    Domain(
        input_features=[if1, if2, if3, if4, if6, If7],
        constraints=[c6, c2],
    ),
    Domain(
        input_features=[if1, if2, if3, if4, if6, If7],
        constraints=[c6, c2, c1],
    ),
]


@pytest.mark.parametrize(
    "domain, candidate_count",
    [
        (domain, candidate_count)
        for domain in domains
        for candidate_count in range(1, 2)
    ],
)
def test_PolytopeSampler(domain, candidate_count):
    sampler = PolytopeSampler(domain=domain)
    samples = sampler.ask(candidate_count)
    if len(domain.constraints.get(NChooseKConstraint)) == 0:
        assert len(samples) == candidate_count


def test_PolytopeSampler_all_fixed():
    domain = Domain(input_features=[if1, if4], constraints=[c5])
    sampler = PolytopeSampler(domain=domain)
    with pytest.warns(UserWarning):
        sampler.ask(2)
