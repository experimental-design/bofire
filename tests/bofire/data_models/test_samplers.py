import pytest

import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs
from bofire.data_models.features.api import CategoricalInput, ContinuousInput

inputs = Inputs(
    features=[ContinuousInput(key=f"if{i}", bounds=(0, 1)) for i in range(1, 4)]
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
        (inputs, constraints, sampling_method, num_samples)
        for sampling_method in ["SOBOL", "UNIFORM", "LHS"]
        for num_samples in [1, 2, 64, 128]
    ],
)
def test_rejection_sampler(features, constraints, sampling_method, num_samples):
    domain = Domain(
        inputs=features,
        constraints=constraints,
    )
    data_model = data_models.RejectionSampler(
        domain=domain, sampling_method=sampling_method
    )
    sampler = strategies.RejectionSampler(data_model=data_model)
    sampler.ask(num_samples)


def test_rejection_sampler_not_converged():
    domain = Domain(
        inputs=inputs,
        constraints=constraints,
    )
    data_model = data_models.RejectionSampler(
        domain=domain, num_base_samples=16, max_iters=2
    )
    sampler = strategies.RejectionSampler(data_model=data_model)
    with pytest.raises(ValueError):
        sampler.ask(128)


if1 = ContinuousInput(
    bounds=(0, 1),
    key="if1",
)
if2 = ContinuousInput(
    bounds=(0, 1),
    key="if2",
)
if3 = ContinuousInput(
    bounds=(0, 1),
    key="if3",
)
if4 = ContinuousInput(
    bounds=(0.1, 0.1),
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
If7 = ContinuousInput(bounds=(1, 1), key="If7")
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
c7 = LinearEqualityConstraint(features=["if1", "if2"], coefficients=[1.0, 1.0], rhs=1.0)

domains = [
    Domain.from_lists(inputs=[if1, if2, if3], constraints=[c2]),
    Domain.from_lists(inputs=[if1, if2, if3, if4], constraints=[c1, c2]),
    Domain.from_lists(inputs=[if1, if2, if3, if4], constraints=[c1, c2, c3]),
    Domain.from_lists(inputs=[if1, if2, if3, if5], constraints=[c2]),
    Domain.from_lists(inputs=[if1, if2, if3, if4, if5], constraints=[c1, c2]),
    Domain.from_lists(inputs=[if1, if2, if3, if4, if5], constraints=[c1, c2, c3]),
    Domain.from_lists(inputs=[if1, if2, if3, if6], constraints=[c2]),
    Domain.from_lists(inputs=[if1, if2, if3, if4, if6], constraints=[c1, c2]),
    Domain.from_lists(inputs=[if1, if2, if3, if4, if6], constraints=[c1, c2, c3]),
    Domain.from_lists(inputs=[if1, if2, if3, if4, if6, If7], constraints=[c1, c2, c3]),
    Domain.from_lists(
        inputs=[if1, if2, if3, if4, if6, If7], constraints=[c1, c2, c3, c4]
    ),
    Domain.from_lists(inputs=[if1, if2, if3, if4], constraints=[c5]),
    Domain.from_lists(
        inputs=[if1, if2, if3, if4, if6, If7],
        constraints=[c6, c2],
    ),
    Domain.from_lists(
        inputs=[if1, if2, if3, if4, if6, If7],
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
    data_model = data_models.PolytopeSampler(domain=domain)
    sampler = strategies.PolytopeSampler(data_model=data_model)
    samples = sampler.ask(candidate_count)
    if len(domain.constraints.get(NChooseKConstraint)) == 0:
        assert len(samples) == candidate_count


def test_PolytopeSampler_interpoint():
    domain = Domain.from_lists(
        inputs=[if1, if2, if3],
        constraints=[InterpointEqualityConstraint(feature="if1", multiplicity=3)],
    )
    data_model = data_models.PolytopeSampler(domain=domain)
    sampler = strategies.PolytopeSampler(data_model=data_model)
    sampler.ask(9)


def test_PolytopeSampler_all_fixed():
    domain = Domain.from_lists(inputs=[if1, if4], constraints=[c5])
    data_model = data_models.PolytopeSampler(domain=domain)
    sampler = strategies.PolytopeSampler(data_model=data_model)
    with pytest.warns(UserWarning):
        sampler.ask(2)


def test_PolytopeSampler_nchoosek():
    domain = Domain.from_lists(
        inputs=[if1, if2, if3, if4, if6, If7],
        constraints=[c6, c2, c7],
    )
    data_model = data_models.PolytopeSampler(domain=domain)
    sampler = strategies.PolytopeSampler(data_model=data_model)
    samples = sampler.ask(5, return_all=True)
    assert len(samples) == 15
    samples = sampler.ask(50, return_all=False)
    assert len(samples) == 50
