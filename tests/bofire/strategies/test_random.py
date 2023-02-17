import warnings

import pytest

from bofire.domain.constraint import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.domain.domain import Domain
from bofire.domain.feature import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.random import RandomStrategy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if0 = ContinuousInput(key="if0", lower_bound=0, upper_bound=1)
if1 = ContinuousInput(key="if1", lower_bound=0, upper_bound=2)
if2 = ContinuousInput(key="if2", lower_bound=0, upper_bound=3)
if3 = CategoricalInput(key="if3", categories=["c1", "c2", "c3"])
if4 = CategoricalInput(
    key="if4", categories=["A", "B", "C"], allowed=[True, True, False]
)
if5 = CategoricalInput(key="if5", categories=["A", "B"], allowed=[True, False])
if6 = CategoricalDescriptorInput(
    key="if6",
    categories=["A", "B", "C"],
    descriptors=["d1", "d2"],
    values=[[1, 2], [3, 7], [5, 1]],
)
if7 = DiscreteInput(key="if7", values=[0, 1, 5])

of1 = ContinuousOutput(key="of1")

c1 = LinearEqualityConstraint(features=["if0", "if1"], coefficients=[1, 1], rhs=1)
c2 = LinearInequalityConstraint(features=["if0", "if1"], coefficients=[1, 1], rhs=1)
c3 = NonlinearEqualityConstraint(expression="if0**2 + if1**2 - 1")
c4 = NonlinearInequalityConstraint(expression="if0**2 + if1**2 - 1")
c5 = NChooseKConstraint(
    features=["if0", "if1", "if2"], min_count=0, max_count=2, none_also_valid=False
)

supported_domains = [
    Domain(
        # continuous features
        input_features=[if0, if1],
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        # continuous features incl. with fixed values
        input_features=[if0, if1, if2],
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        # all feature types
        input_features=[if1, if3, if6, if7],
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        # all feature types incl. with fixed values
        input_features=[if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[],
    ),
    Domain(
        # all feature types, linear equality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c1],
    ),
    Domain(
        # all feature types, linear inequality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c2],
    ),
    Domain(
        # all feature types, nonlinear inequality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c4],
    ),
]

unsupported_domains = [
    Domain(
        # nonlinear equality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c3],
    ),
    Domain(
        # combination of linear equality and nonlinear inequality
        input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
        output_features=[of1],
        constraints=[c1, c4],
    ),
    # Domain(
    #     # n-choose-k
    #     input_features=[if0, if1, if2, if3, if4, if5, if6, if7],
    #     output_features=[of1],
    #     constraints=[c5],
    # ),
]


@pytest.mark.parametrize("domain", supported_domains)
def test_ask(domain):
    strategy = RandomStrategy(domain=domain)
    candidates = strategy.ask(3)
    assert len(candidates) == 3


@pytest.mark.parametrize("domain", unsupported_domains)
def test_unsupported(domain):
    with pytest.raises(Exception):
        RandomStrategy(domain=domain)
