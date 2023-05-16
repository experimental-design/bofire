import warnings

import pytest

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
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.api import RandomStrategy

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if0 = ContinuousInput(key="if0", bounds=(0, 1))
if1 = ContinuousInput(key="if1", bounds=(0, 2))
if2 = ContinuousInput(key="if2", bounds=(0, 3))
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
    Domain.from_lists(
        # continuous features
        inputs=[if0, if1],
        outputs=[of1],
        constraints=[],
    ),
    Domain.from_lists(
        # continuous features incl. with fixed values
        inputs=[if0, if1, if2],
        outputs=[of1],
        constraints=[],
    ),
    Domain.from_lists(
        # all feature types
        inputs=[if1, if3, if6, if7],
        outputs=[of1],
        constraints=[],
    ),
    Domain.from_lists(
        # all feature types incl. with fixed values
        inputs=[if1, if2, if3, if4, if5, if6, if7],
        outputs=[of1],
        constraints=[],
    ),
    Domain.from_lists(
        # all feature types, linear equality
        inputs=[if0, if1, if2, if3, if4, if5, if6, if7],
        outputs=[of1],
        constraints=[c1],
    ),
    Domain.from_lists(
        # all feature types, linear inequality
        inputs=[if0, if1, if2, if3, if4, if5, if6, if7],
        outputs=[of1],
        constraints=[c2],
    ),
    Domain.from_lists(
        # all feature types, nonlinear inequality
        inputs=[if0, if1, if2, if3, if4, if5, if6, if7],
        outputs=[of1],
        constraints=[c4],
    ),
]

unsupported_domains = [
    Domain.from_lists(
        # nonlinear equality
        inputs=[if0, if1, if2, if3, if4, if5, if6, if7],
        outputs=[of1],
        constraints=[c3],
    ),
    Domain.from_lists(
        # combination of linear equality and nonlinear inequality
        inputs=[if0, if1, if2, if3, if4, if5, if6, if7],
        outputs=[of1],
        constraints=[c1, c4],
    ),
    # Domain(
    #     # n-choose-k
    #     inputs=[if0, if1, if2, if3, if4, if5, if6, if7],
    #     outputs=[of1],
    #     constraints=[c5],
    # ),
]


@pytest.mark.parametrize("domain", supported_domains)
def test_ask(domain):
    data_model = data_models.RandomStrategy(domain=domain)
    strategy = strategies.map(data_model=data_model)
    candidates = strategy.ask(3)
    assert len(candidates) == 3


@pytest.mark.parametrize("domain", unsupported_domains)
def test_unsupported(domain):
    with pytest.raises(Exception):
        data_model = data_models.RandomStrategy(domain=domain)
        RandomStrategy(data_model=data_model)
