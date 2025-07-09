import warnings

import pytest
from pandas.testing import assert_frame_equal

import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.data_models.constraints.api import (
    CategoricalExcludeConstraint,
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    SelectionCondition,
    ThresholdCondition,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if0 = ContinuousInput(key="if0", bounds=(0, 1))
if1 = ContinuousInput(key="if1", bounds=(0, 2))
if2 = ContinuousInput(key="if2", bounds=(0, 3))
if3 = CategoricalInput(key="if3", categories=["c1", "c2", "c3"])
if4 = CategoricalInput(
    key="if4",
    categories=["A", "B", "C"],
    allowed=[True, True, False],
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
c3 = NonlinearEqualityConstraint(
    expression="if0**2 + if1**2 - 1", features=["if0", "if1"]
)
c4 = NonlinearInequalityConstraint(
    expression="if0**2 + if1**2 - 1", features=["if0", "if1"]
)
c5 = NChooseKConstraint(
    features=["if0", "if1", "if2"],
    min_count=0,
    max_count=2,
    none_also_valid=False,
)
c6 = CategoricalExcludeConstraint(
    features=["if4", "if2"],
    conditions=[
        SelectionCondition(selection=["A"]),
        ThresholdCondition(threshold=0.5, operator=">"),
    ],
    logical_op="AND",
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
        # combination of linear equality and nonlinear inequality
        inputs=[if0, if1, if2, if3, if4, if5, if6, if7],
        outputs=[of1],
        constraints=[c1, c4],
    ),
    Domain.from_lists(
        # all ordered feature types, non-linear inequality
        inputs=[if0, if1, if2, if7],
        outputs=[of1],
        constraints=[c4],
    ),
    Domain.from_lists(inputs=[if2, if4], constraints=[c6], outputs=[of1]),
]


@pytest.mark.parametrize("domain", supported_domains)
def test_ask(domain):
    data_model = data_models.RandomStrategy(domain=domain)
    strategy = strategies.map(data_model=data_model)
    candidates = strategy.ask(3)
    assert len(candidates) == 3
    assert domain.constraints.is_fulfilled(candidates).all()


def test_rejection_sampler_not_converged():
    data_model = data_models.RandomStrategy(
        domain=supported_domains[-2],
        num_base_samples=4,
        max_iters=2,
    )
    sampler = strategies.RandomStrategy(data_model=data_model)
    with pytest.raises(
        ValueError,
        match="Maximum iterations exceeded in rejection sampling.",
    ):
        sampler.ask(128)


def test_interpoint():
    domain = Domain.from_lists(
        inputs=[if1, if2, if3],
        constraints=[InterpointEqualityConstraint(features=["if1"], multiplicity=3)],
    )
    data_model = data_models.RandomStrategy(domain=domain)
    sampler = strategies.RandomStrategy(data_model=data_model)
    sampler.ask(9)


def test_all_fixed():
    if1 = ContinuousInput(
        bounds=(0, 1),
        key="if1",
    )
    if4 = ContinuousInput(
        bounds=(0.1, 0.1),
        key="if4",
    )
    domain = Domain.from_lists(
        inputs=[if1, if4],
        constraints=[
            LinearEqualityConstraint(
                features=["if1", "if4"],
                coefficients=[1.0, 1.0],
                rhs=1.0,
            ),
        ],
    )
    data_model = data_models.RandomStrategy(domain=domain)
    sampler = strategies.RandomStrategy(data_model=data_model)
    with pytest.warns(UserWarning):
        sampler.ask(2)


def test_nchoosek():
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

    if6 = CategoricalInput(
        categories=["a", "b", "c"],
        allowed=[False, True, False],
        key="if6",
    )
    If7 = ContinuousInput(bounds=(1, 1), key="If7")

    c2 = LinearInequalityConstraint.from_greater_equal(
        features=["if1", "if2"],
        coefficients=[1.0, 1.0],
        rhs=0.2,
    )

    c6 = NChooseKConstraint(
        features=["if1", "if2", "if3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    c7 = LinearEqualityConstraint(
        features=["if1", "if2"],
        coefficients=[1.0, 1.0],
        rhs=1.0,
    )
    domain = Domain.from_lists(
        inputs=[if1, if2, if3, if4, if6, If7],
        constraints=[c6, c2, c7],
    )
    data_model = data_models.RandomStrategy(domain=domain)
    sampler = strategies.RandomStrategy(data_model=data_model)
    samples = sampler.ask(50)
    assert len(samples) == 50


def test_sample_from_polytope():
    if1 = ContinuousInput(
        bounds=(0, 1),
        key="if1",
    )
    if2 = ContinuousInput(
        bounds=(0, 1),
        key="if2",
    )
    c2 = LinearInequalityConstraint.from_greater_equal(
        features=["if1", "if2"],
        coefficients=[1.0, 1.0],
        rhs=0.8,
    )
    domain = Domain.from_lists(
        inputs=[if1, if2],
        constraints=[c2],
    )
    samples = strategies.RandomStrategy._sample_from_polytope(domain, 5)
    samples2 = strategies.RandomStrategy._sample_from_polytope(domain, 5, seed=42)
    samples3 = strategies.RandomStrategy._sample_from_polytope(domain, 5, seed=42)
    assert_frame_equal(samples2, samples3)
    with pytest.raises(AssertionError):
        assert_frame_equal(samples2, samples)
