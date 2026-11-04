import warnings

import pytest
from botorch.exceptions.errors import InfeasibilityError
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
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.features.descriptors import Descriptors


@pytest.mark.parametrize(
    "free_count", [0, 1], ids=["all-pinned-to-zero", "one-free-capped-at-0.6"]
)
def test_polytope_infeasible_pinned_mixture(free_count):
    """Issue #842: zero or one unpinned component cannot sum to one.

    free_count counts components allowed in [0, 0.6]; all others are fixed at zero.
    Zero tests a constant contradiction; one tests an out-of-bounds implied value.
    """
    keys = ["x0", "x1", "x2"]
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key=k, bounds=(0, 0.6) if i < free_count else (0, 0))
            for i, k in enumerate(keys)
        ],
        constraints=[
            LinearEqualityConstraint(features=keys, coefficients=[1] * 3, rhs=1)
        ],
    )
    with pytest.raises(InfeasibilityError):
        strategies.RandomStrategy._sample_from_polytope(domain=domain, n=2, seed=1)


@pytest.mark.parametrize("weighted", [False, True])
def test_nchoosek_mixture_replenishes_samples(monkeypatch, weighted):
    """Issue #842: rejected subsets must not reduce ask's requested row count.

    The weighted case also preserves the surviving subsets' 3:1 multiplicity.
    """
    keys = [f"x{i}" for i in range(4)]
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=k, bounds=(0, 0.6)) for k in keys],
        constraints=[
            NChooseKConstraint(
                features=keys, min_count=0, max_count=2, none_also_valid=False
            ),
            LinearEqualityConstraint(features=keys, coefficients=[1] * 4, rhs=1),
        ],
    )
    strategy = strategies.RandomStrategy(
        data_model=data_models.RandomStrategy(
            domain=domain, seed=1, n_burnin=10, n_thinning=1
        )
    )
    monkeypatch.setattr(
        strategy,
        "sample_valid_nchoosek_features",
        lambda **kwargs: (
            [(), ("x0", "x1"), ("x0", "x1"), ("x0", "x1"), ("x2", "x3")]
            if weighted
            else [(), ("x0",), ("x0", "x1")]
        ),
    )
    before = domain.model_dump()
    requested = 20 if weighted else 12
    candidates = strategy.ask(requested)
    assert len(candidates) == requested
    if weighted:
        assert (candidates["x0"] > 0).sum() == 15
        assert (candidates["x2"] > 0).sum() == 5
    domain.validate_candidates(candidates, only_inputs=True)
    assert domain.model_dump() == before


@pytest.mark.parametrize(
    "conflict", ["bounds", "equalities", "inequality", "zero_coefficient"]
)
def test_polytope_fixed_constraint_conflicts(conflict):
    """Issue #842: substituting fixed values must reveal infeasible constraints.

    With b fixed at zero and a bounded by [0, 1], the cases reduce to:
    bounds: a=2, outside its bounds;
    equalities: a=0.5 and a=0.25, contradictory fixed values;
    inequality: a=0.5 and a<=0.25, contradictory requirements;
    zero_coefficient: a=0.5 and 0*a+0=1, an impossible constant equality.

    Each must raise InfeasibilityError rather than return invalid samples, so
    NChooseK sampling can recognize and skip the infeasible subset.
    """
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="a", bounds=(0, 1)),
            ContinuousInput(key="b", bounds=(0, 0)),
        ],
        constraints=[
            LinearEqualityConstraint(
                features=["a", "b"],
                coefficients=[1, 1],
                rhs=2 if conflict == "bounds" else 0.5,
            ),
        ],
    )
    if conflict == "equalities":
        domain.constraints.constraints.append(
            LinearEqualityConstraint(features=["a", "b"], coefficients=[1, 1], rhs=0.25)
        )
    elif conflict == "inequality":
        domain.constraints.constraints.append(
            LinearInequalityConstraint(
                features=["a", "b"], coefficients=[1, 1], rhs=0.25
            )
        )
    elif conflict == "zero_coefficient":
        domain.constraints.constraints.append(
            LinearEqualityConstraint(features=["a", "b"], coefficients=[0, 1], rhs=1)
        )
    with pytest.raises(InfeasibilityError):
        strategies.RandomStrategy._sample_from_polytope(domain=domain, n=2, seed=1)


def test_polytope_chained_pseudo_fixed():
    """Issue #842: propagate implied fixed values through successive equalities.

    With b fixed at zero, a+b=0.4 first fixes a=0.4; substituting that into
    a+c=1 then fixes c=0.6. Reduction must continue until no further values
    are implied, rather than stopping after checking each equality once.
    Both requested rows must contain these fixed values and satisfy the domain.
    """
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="a", bounds=(0, 1)),
            ContinuousInput(key="b", bounds=(0, 0)),
            ContinuousInput(key="c", bounds=(0, 1)),
        ],
        constraints=[
            LinearEqualityConstraint(features=["a", "b"], coefficients=[1, 1], rhs=0.4),
            LinearEqualityConstraint(features=["a", "c"], coefficients=[1, 1], rhs=1),
        ],
    )
    candidates = strategies.RandomStrategy._sample_from_polytope(
        domain=domain, n=2, seed=1
    )
    domain.validate_candidates(candidates, only_inputs=True)
    assert candidates["a"].tolist() == [0.4, 0.4]
    assert candidates["c"].tolist() == [0.6, 0.6]


def test_polytope_pseudo_fixed_decimal_boundary():
    """Bounds validation must not reject a feasible decimal equality through rounding.

    With b=0, 0.1*a+b=0.3 fixes a at its lower bound of 3. Converting 0.1
    through float32 first moves the computed value below that bound; even
    double-precision division needs allowance for floating-point roundoff.
    """
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="a", bounds=(3, 4)),
            ContinuousInput(key="b", bounds=(0, 0)),
        ],
        constraints=[
            LinearEqualityConstraint(
                features=["a", "b"], coefficients=[0.1, 1], rhs=0.3
            )
        ],
    )
    candidates = strategies.RandomStrategy._sample_from_polytope(
        domain=domain, n=2, seed=1
    )
    assert candidates["a"].tolist() == [3, 3]
    domain.validate_candidates(candidates, only_inputs=True)


@pytest.mark.parametrize(
    "recover", [True, False], ids=["later-draw-recovers", "budget-exhausted"]
)
def test_nchoosek_retries_infeasible_subset_draws(monkeypatch, recover):
    """Issue #842: retry unlucky subset draws before reporting budget exhaustion.

    Empty and singleton subsets cannot satisfy a+b=1 with component caps of 0.6.
    Return those twice, then offer the feasible pair on the final allowed draw
    in the recovery case. Otherwise exhaust all three draws and report a sampled
    search failure, not proof that the domain is infeasible. Previously rejected
    subsets must not invoke the polytope sampler again within this ask call.
    """
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=k, bounds=(0, 0.6)) for k in ["a", "b"]],
        constraints=[
            NChooseKConstraint(
                features=["a", "b"], min_count=0, max_count=2, none_also_valid=False
            ),
            LinearEqualityConstraint(features=["a", "b"], coefficients=[1, 1], rhs=1),
        ],
    )
    strategy = strategies.RandomStrategy(
        data_model=data_models.RandomStrategy(
            domain=domain, seed=1, max_iters=3, n_burnin=10, n_thinning=1
        )
    )
    draws = 0
    sampled = []
    polytope_sampler = strategy._sample_from_polytope

    def draw_subsets(**kwargs):
        nonlocal draws
        draws += 1
        return [("a", "b"), ("a",)] if recover and draws == 3 else [(), ("a",)]

    def record_polytope_call(**kwargs):
        sampled.append(
            tuple(
                feature.key
                for feature in kwargs["domain"].inputs.get(ContinuousInput)
                if not feature.is_fixed()
            )
        )
        return polytope_sampler(**kwargs)

    monkeypatch.setattr(strategy, "sample_valid_nchoosek_features", draw_subsets)
    monkeypatch.setattr(strategy, "_sample_from_polytope", record_polytope_call)
    before = domain.model_dump()
    if recover:
        candidates = strategy.ask(2)
        assert len(candidates) == 2
        domain.validate_candidates(candidates, only_inputs=True)
    else:
        with pytest.raises(ValueError, match="after 3 subset-draw batches"):
            strategy.ask(2)
    assert draws == 3
    assert sampled.count(()) == 1
    assert sampled.count(("a",)) == 1
    # The feasible pair supplies one initial row and one fresh replenishment row.
    assert sampled.count(("a", "b")) == (2 if recover else 0)
    assert domain.model_dump() == before


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_nchoosek_single_candidate_retries(seed):
    """Issue #842: a one-row request must find the feasible pair despite unlucky draws.

    Only one subset is drawn per batch for ask(1); real seeded draws exercise
    recovery without requiring the caller to increase the requested row count.
    """
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=k, bounds=(0, 0.6)) for k in ["a", "b"]],
        constraints=[
            NChooseKConstraint(
                features=["a", "b"], min_count=0, max_count=2, none_also_valid=False
            ),
            LinearEqualityConstraint(features=["a", "b"], coefficients=[1, 1], rhs=1),
        ],
    )
    strategy = strategies.RandomStrategy(
        data_model=data_models.RandomStrategy(
            domain=domain, seed=seed, max_iters=30, n_burnin=10, n_thinning=1
        )
    )
    candidates = strategy.ask(1)
    assert len(candidates) == 1
    domain.validate_candidates(candidates, only_inputs=True)


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
if6 = CategoricalInput(
    key="if6",
    categories=["A", "B", "C"],
    descriptors=Descriptors(columns={"d1": [1, 3, 5], "d2": [2, 7, 1]}),
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
        features=["if1", "if2", "if3"],
        coefficients=[1.0, 1.0, 1.0],
        rhs=0.2,
    )

    c6 = NChooseKConstraint(
        features=["if1", "if2", "if3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    c7 = LinearEqualityConstraint(
        features=["if1", "if2", "if3"],
        coefficients=[1.0, 1.0, 1.0],
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


def test_allow_zero_without_nchoosek():
    """Test random sampling with allow_zero features but no NChooseK constraint."""
    if1 = ContinuousInput(bounds=(0.1, 1), key="if1", allow_zero=True)
    if2 = ContinuousInput(bounds=(0.1, 1), key="if2", allow_zero=True)
    if3 = ContinuousInput(bounds=(0.1, 1), key="if3")
    domain = Domain.from_lists(inputs=[if1, if2, if3])
    data_model = data_models.RandomStrategy(domain=domain)
    sampler = strategies.RandomStrategy(data_model=data_model)
    samples = sampler.ask(50)
    assert len(samples) == 50
    # if3 should never be zero (not allow_zero)
    assert (samples["if3"] != 0.0).all()
    # if1 and if2 should have some zeros (allow_zero)
    assert (samples["if1"] == 0.0).any() or (samples["if2"] == 0.0).any()


def test_allow_zero_with_nchoosek():
    """Test that allow_zero features already in NChooseK don't get duplicate groups."""
    if1 = ContinuousInput(bounds=(0, 1), key="if1")
    if2 = ContinuousInput(bounds=(0, 1), key="if2")
    if3 = ContinuousInput(bounds=(0, 1), key="if3")
    if4 = ContinuousInput(bounds=(0.1, 1), key="if4", allow_zero=True)
    c = NChooseKConstraint(
        features=["if1", "if2", "if3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    domain = Domain.from_lists(inputs=[if1, if2, if3, if4], constraints=[c])
    data_model = data_models.RandomStrategy(domain=domain)
    sampler = strategies.RandomStrategy(data_model=data_model)
    samples = sampler.ask(50)
    assert len(samples) == 50
    # At most 2 features should be non-zero per sample (from NChooseK)
    nonzero_counts = (samples[["if1", "if2", "if3"]] != 0.0).sum(axis=1)
    assert (nonzero_counts >= 1).all()
    assert (nonzero_counts <= 2).all()
    # if4 (allow_zero, not in NChooseK) should have some zeros
    assert (samples["if4"] == 0.0).any()


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


@pytest.mark.parametrize(
    "method,kwargs,n_samples",
    [
        (SamplingMethodEnum.SOBOL, {"scramble": True}, 10),
        (SamplingMethodEnum.SOBOL, {"scramble": False}, 10),
        (SamplingMethodEnum.LHS, {"scramble": True, "strength": 1}, 10),
        (
            SamplingMethodEnum.LHS,
            {"strength": 2},
            9,
        ),  # strength=2 requires n to be square of prime
        (SamplingMethodEnum.UNIFORM, {}, 10),
    ],
)
def test_sampler_kwargs_various_methods(method, kwargs, n_samples):
    """Test sampler_kwargs with various sampling methods."""
    test_domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 2)),
                ContinuousInput(key="x3", bounds=(0, 3)),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
    )
    sampler_data_model = data_models.RandomStrategy(
        domain=test_domain, fallback_sampling_method=method, sampler_kwargs=kwargs
    )
    sampler_strategy = strategies.RandomStrategy(data_model=sampler_data_model)
    candidates = sampler_strategy.ask(n_samples)
    assert len(candidates) == n_samples


def test_sample_valid_nchoosek_features_uniform_over_subsets():
    """With one NChooseK on n=5 features and k in [1, 3], there are
    C(5,1)+C(5,2)+C(5,3) = 25 valid subsets. With uniform sampling each
    should appear with frequency ~1/25.
    """
    inputs = [ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(5)]
    constraint = NChooseKConstraint(
        features=[f"x{i}" for i in range(5)],
        min_count=1,
        max_count=3,
        none_also_valid=False,
    )
    domain = Domain.from_lists(inputs=inputs, constraints=[constraint])
    n_samples = 25_000
    samples = strategies.RandomStrategy.sample_valid_nchoosek_features(
        domain=domain, seed=0, n=n_samples
    )
    counts: dict = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    assert len(counts) == 25, f"Expected 25 unique subsets, got {len(counts)}"
    expected = n_samples / 25
    for subset, count in counts.items():
        rel = abs(count - expected) / expected
        assert (
            rel < 0.20
        ), f"Subset {subset} count {count} too far from expected {expected:.0f}"


def test_sample_valid_nchoosek_features_none_also_valid():
    """When none_also_valid=True, the empty subset is in the support."""
    inputs = [ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(3)]
    constraint = NChooseKConstraint(
        features=["x0", "x1", "x2"],
        min_count=2,
        max_count=3,
        none_also_valid=True,
    )
    domain = Domain.from_lists(inputs=inputs, constraints=[constraint])
    samples = strategies.RandomStrategy.sample_valid_nchoosek_features(
        domain=domain, seed=1, n=2000
    )
    unique = set(samples)
    # Valid subsets: () + C(3,2) + C(3,3) = 1 + 3 + 1 = 5
    assert len(unique) == 5
    assert () in unique


def test_sample_valid_nchoosek_features_allow_zero_singletons():
    """Without any NChooseK, allow_zero=True features form singleton groups."""
    inputs = [
        ContinuousInput(key="a", bounds=(0.1, 1), allow_zero=True),
        ContinuousInput(key="b", bounds=(0.1, 1), allow_zero=True),
        ContinuousInput(key="c", bounds=(0.1, 1)),
    ]
    domain = Domain.from_lists(inputs=inputs)
    samples = strategies.RandomStrategy.sample_valid_nchoosek_features(
        domain=domain, seed=2, n=2000
    )
    unique = set(samples)
    # Each of {a, b} can be on or off independently -> 4 subsets
    assert unique == {(), ("a",), ("b",), ("a", "b")}


def test_sample_valid_nchoosek_features_empty_returns_empty_tuple():
    """Domain without NChooseK and without allow_zero features yields ()."""
    inputs = [ContinuousInput(key="x", bounds=(0, 1))]
    domain = Domain.from_lists(inputs=inputs)
    samples = strategies.RandomStrategy.sample_valid_nchoosek_features(
        domain=domain, seed=3, n=4
    )
    assert samples == [(), (), (), ()]
