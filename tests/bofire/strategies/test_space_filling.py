import pytest
from pandas import concat

import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.strategies.doe import SpaceFillingCriterion


pytest.importorskip("cyipopt")

inputs = [ContinuousInput(key=f"if{i}", bounds=(0, 1)) for i in range(1, 4)]
c1 = LinearInequalityConstraint(
    features=["if1", "if2", "if3"],
    coefficients=[1, 1, 1],
    rhs=1,
)
c2 = LinearEqualityConstraint(
    features=["if1", "if2", "if3"],
    coefficients=[1, 1, 1],
    rhs=1,
)
c3 = NonlinearEqualityConstraint(
    expression="if1**2 + if2**2 - if3",
    features=["if1", "if2", "if3"],
)
c4 = NonlinearInequalityConstraint(
    expression="if1**2 + if2**2 - if3",
    features=["if1", "if2", "if3"],
)
c5 = NChooseKConstraint(
    features=["if1", "if2", "if3"],
    min_count=0,
    max_count=1,
    none_also_valid=True,
)


domains = [
    Domain.from_lists(inputs=inputs, constraints=[c1]),
    Domain.from_lists(inputs=inputs, constraints=[c2]),
    Domain.from_lists(inputs=inputs, constraints=[c3]),
    Domain.from_lists(inputs=inputs, constraints=[c4]),
    Domain.from_lists(inputs=inputs, constraints=[c5]),
]


@pytest.mark.parametrize(
    "domain, num_samples",
    [(domain, candidate_count) for domain in domains for candidate_count in [1, 16]],
)
def test_ask(domain, num_samples):
    data_model = data_models.DoEStrategy(
        domain=domain,
        criterion=SpaceFillingCriterion(),
        ipopt_options={"max_iter": 300, "print_level": 0},
    )
    sampler = strategies.DoEStrategy(data_model=data_model)
    samples = sampler.ask(num_samples)
    assert len(samples) == num_samples


test_ask(domain=domains[0], num_samples=1)


def test_ask_pending_candidates():
    data_model = data_models.DoEStrategy(
        domain=domains[0],
        criterion=SpaceFillingCriterion(),
        ipopt_options={"max_iter": 300, "print_level": 0},
    )
    sampler = strategies.DoEStrategy(data_model=data_model)
    pending_candidates = sampler.ask(2, add_pending=True)
    samples = sampler.ask(1)
    assert len(samples) == 1
    all_samples = concat(
        [samples, pending_candidates],
        axis=0,
        ignore_index=True,
    ).drop_duplicates()
    assert len(all_samples) == 3
