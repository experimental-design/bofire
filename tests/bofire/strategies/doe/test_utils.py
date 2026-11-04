import importlib.util
import sys
from math import comb

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, NonlinearConstraint, OptimizeResult

from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.doe import utils as doe_utils
from bofire.strategies.doe.objective import DOptimalityCriterion, get_objective_function
from bofire.strategies.doe.utils import (
    ConstraintWrapper,
    _build_nchoosek_combined_patterns,
    _minimize,
    constraints_as_scipy_constraints,
    convert_formula_to_string,
    formula_str_to_fully_continuous,
    get_formula_from_string,
    n_zero_eigvals,
    nchoosek_constraints_as_bounds,
)


CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None


# Infeasible-pattern errors read: "No combined NChooseK patterns are feasible
# under the input bounds and linear constraints." The pytest.raises match below
# checks the key terms "input bounds and linear constraints", which identify the
# failure without coupling tests to the entire sentence's wording.


@pytest.mark.parametrize("max_count", [2, 3])
@pytest.mark.parametrize(
    "upper_bound, minimum_required",
    [(0.3, 4), (0.4, 3), (0.49, 3), (0.5, 2), (0.6, 2), (0.99, 2), (1.0, 1)],
)
def test_nchoosek_mixture_patterns(max_count, upper_bound, minimum_required):
    """Issue #842: filter by component caps without modifying the domain.

    Requesting 20 experiment slots forces the assignment to cycle through the
    surviving patterns (6 for max_count=2 or 10 for max_count=3 when 0.5 <= cap < 1).
    Caps 0.3 and 0.4 require four and three components respectively, testing empty
    and reduced feasible sets; 0.49/0.5 straddle pair feasibility, 0.6 reproduces
    the issue, and 0.99/1.0 straddle singleton feasibility for a mixture sum of one.
    """
    keys = [f"x{i}" for i in range(4)]
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=k, bounds=(0, upper_bound)) for k in keys],
        constraints=[
            NChooseKConstraint(
                features=keys, min_count=0, max_count=max_count, none_also_valid=False
            ),
            LinearEqualityConstraint(features=keys, coefficients=[1] * 4, rhs=1),
        ],
    )
    before = domain.model_dump()
    expected_count = sum(comb(4, k) for k in range(minimum_required, max_count + 1))
    for _ in range(2):
        if expected_count == 0:
            with pytest.raises(ValueError, match="input bounds and linear constraints"):
                _build_nchoosek_combined_patterns(domain)
            with pytest.raises(ValueError, match="input bounds and linear constraints"):
                nchoosek_constraints_as_bounds(domain, 20)
            assert domain.model_dump() == before
            continue
        patterns = _build_nchoosek_combined_patterns(domain)
        assert len(patterns) == expected_count
        assert all(
            minimum_required <= 4 - len(pattern) <= max_count for pattern in patterns
        )
        bounds = np.array(nchoosek_constraints_as_bounds(domain, 20)).reshape(20, 4, 2)
        assert (bounds[:, :, 1].sum(axis=1) >= 1).all()
        assert domain.model_dump() == before


@pytest.mark.parametrize(
    "scenario", ["inequality", "joint", "all_features_unpinned", "none"]
)
def test_nchoosek_linear_filter(scenario):
    """Issue #842: check linear constraints jointly, preserving unconstrained patterns."""
    keys = ["a", "b", "c"]
    constraints = [
        NChooseKConstraint(
            features=keys,
            min_count=3 if scenario == "all_features_unpinned" else 1,
            max_count=3 if scenario == "all_features_unpinned" else 2,
            none_also_valid=False,
        )
    ]
    if scenario == "inequality":
        constraints.append(
            LinearInequalityConstraint(
                features=["a", "b"], coefficients=[-1, -1], rhs=-0.5
            )
        )
    elif scenario == "joint":
        constraints.extend(
            [
                LinearEqualityConstraint(
                    features=["a", "b"], coefficients=[1, 1], rhs=1
                ),
                LinearInequalityConstraint(
                    features=["a", "b"], coefficients=[1, 1], rhs=0.5
                ),
            ]
        )
    elif scenario == "all_features_unpinned":
        constraints.append(
            LinearEqualityConstraint(features=keys, coefficients=[1] * 3, rhs=1)
        )
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=k, bounds=(0, 1)) for k in keys],
        constraints=constraints,
    )
    if scenario == "joint":
        with pytest.raises(ValueError, match="input bounds and linear constraints"):
            _build_nchoosek_combined_patterns(domain)
    else:
        patterns = set(_build_nchoosek_combined_patterns(domain))
        if scenario == "all_features_unpinned":
            assert patterns == {()}
        else:
            expected = {(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)}
            if scenario == "inequality":
                expected.remove((0, 1))
            assert patterns == expected


@pytest.mark.parametrize(
    "rhs, expected_inactive",
    [
        pytest.param(1.0, {("z",)}, id="one-pattern-one-inactive"),
        pytest.param(
            0.75, {("z",), ("b",), ("b", "z")}, id="multiple-patterns-and-inactive"
        ),
    ],
)
def test_nchoosek_filter_feature_order(rhs, expected_inactive):
    """Issue #842 discussion: shuffled inputs must still pin the correct named features.

    Asymmetric bounds and reordered coefficients expose index mix-ups; the second
    case checks multiple surviving patterns, including two deactivated features.
    """
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="z", bounds=(0, 0.1)),
            ContinuousInput(key="b", bounds=(0, 0.3)),
            ContinuousInput(key="a", bounds=(0, 0.8)),
        ],
        constraints=[
            NChooseKConstraint(
                features=["z", "a", "b"],
                min_count=1,
                max_count=2,
                none_also_valid=False,
            ),
            LinearEqualityConstraint(
                features=["b", "z", "a"], coefficients=[2, 1, 1], rhs=rhs
            ),
        ],
    )
    keys = domain.inputs.get_keys()
    assert keys == ["a", "b", "z"]
    patterns = _build_nchoosek_combined_patterns(domain)
    # ("z",) means only z is deactivated; a and b remain unpinned.
    assert {tuple(keys[idx] for idx in p) for p in patterns} == expected_inactive

    # Request twice as many experiments as patterns; each pattern must appear twice.
    n_experiments = 2 * len(expected_inactive)
    bounds = np.array(nchoosek_constraints_as_bounds(domain, n_experiments)).reshape(
        n_experiments, 3, 2
    )
    expected_upper = {
        tuple(0 if key in inactive else cap for key, cap in zip(keys, [0.8, 0.3, 0.1]))
        for inactive in expected_inactive
    }
    observed_upper = [tuple(row) for row in bounds[:, :, 1]]
    assert set(observed_upper) == expected_upper
    assert all(observed_upper.count(row) == 2 for row in expected_upper)
    np.testing.assert_array_equal(bounds[:, :, 0], np.zeros((n_experiments, 3)))


@pytest.mark.parametrize("scenario", ["equality", "inequality"])
def test_nchoosek_filter_signed_coefficients(scenario):
    """Issue #842: signed coefficients must leave only the pattern that pins a.

    For a-2*b = -1 (or <= -1), a=0 permits b=0.5, but b=0 requires
    a=-1 (or <= -1), outside its bounds; coefficients are supplied in b,a order.
    """
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="a", bounds=(0, 0.4)),
            ContinuousInput(key="b", bounds=(0, 0.6)),
        ],
        constraints=[
            NChooseKConstraint(
                features=["a", "b"], min_count=1, max_count=1, none_also_valid=False
            ),
            (
                LinearEqualityConstraint
                if scenario == "equality"
                else LinearInequalityConstraint
            )(features=["b", "a"], coefficients=[-2, 1], rhs=-1),
        ],
    )
    # Canonical feature order is a,b: [(0,)] means only pinning a survives.
    assert _build_nchoosek_combined_patterns(domain) == [(0,)]


@pytest.mark.parametrize(
    "nchoosek, linear", [(False, False), (False, True), (True, False)]
)
def test_nchoosek_filter_skips_lp(nchoosek, linear, monkeypatch):
    """Issue #842: avoid solver overhead when no joint feasibility check is needed.

    With no NChooseK patterns or no linear constraints, filtering must never call
    the LP solver; a replacement that fails on any call enforces this.
    """
    keys = ["a", "b"]
    constraints = []
    if nchoosek:
        constraints.append(
            NChooseKConstraint(
                features=keys, min_count=1, max_count=2, none_also_valid=False
            )
        )
    if linear:
        constraints.append(
            LinearEqualityConstraint(features=keys, coefficients=[1, 1], rhs=1)
        )
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=k, bounds=(0, 1)) for k in keys],
        constraints=constraints,
    )

    def unexpected_lp(*args, **kwargs):
        pytest.fail("LP called on a fast path")

    monkeypatch.setattr(doe_utils.opt, "linprog", unexpected_lp)
    assert len(_build_nchoosek_combined_patterns(domain)) == (3 if nchoosek else 0)


def test_nchoosek_filter_checks_each_unique_pattern_once(monkeypatch):
    """Issue #842: deduplicate patterns before paying for joint LP checks.

    The substituted generator returns pin-a twice and pin-b once: only two
    unique patterns need checking, even when reused for 20 experiments.
    Two linear constraints exercise the joint LP path rather than interval-only
    filtering; 40 returned bound pairs represent 20 experiments with two features.
    """
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=k, bounds=(0, 1)) for k in ["a", "b"]],
        constraints=[
            LinearEqualityConstraint(features=["a", "b"], coefficients=[1, 1], rhs=1),
            LinearInequalityConstraint(
                features=["a", "b"], coefficients=[-1, -1], rhs=0
            ),
        ],
    )
    monkeypatch.setattr(
        doe_utils, "_get_nchoosek_combined_patterns", lambda _: [(0,), (0,), (1,)]
    )
    linprog = doe_utils.opt.linprog
    calls = 0

    def counted_lp(*args, **kwargs):
        nonlocal calls
        calls += 1
        return linprog(*args, **kwargs)

    monkeypatch.setattr(doe_utils.opt, "linprog", counted_lp)
    assert len(nchoosek_constraints_as_bounds(domain, 20)) == 40
    assert calls == 2


def test_nchoosek_filter_surfaces_solver_failure(monkeypatch):
    """Issue #842: a numerical solver failure must not discard a pattern silently.

    Both pin-a and pin-b can satisfy a+b=1 and a+b>=0 within their bounds.
    A substituted LP result reports numerical failure, not proven infeasibility,
    so filtering must raise RuntimeError containing the solver's explanation.
    """
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=k, bounds=(0, 1)) for k in ["a", "b"]],
        constraints=[
            NChooseKConstraint(
                features=["a", "b"], min_count=1, max_count=1, none_also_valid=False
            ),
            LinearEqualityConstraint(features=["a", "b"], coefficients=[1, 1], rhs=1),
            LinearInequalityConstraint(
                features=["a", "b"], coefficients=[-1, -1], rhs=0
            ),
        ],
    )

    def failed_lp(*args, **kwargs):
        return OptimizeResult(success=False, status=4, message="numerical failure")

    monkeypatch.setattr(doe_utils.opt, "linprog", failed_lp)
    with pytest.raises(RuntimeError, match="numerical failure"):
        _build_nchoosek_combined_patterns(domain)


@pytest.mark.parametrize("n_experiments", [5, 6, 8])
def test_doe_mixture_public_ask(n_experiments):
    """Issue #842: strategy.ask(n_experiments) must return the requested valid rows.

    Requests of 5, 6, and 8 are below, equal to, and above the six feasible
    pair patterns; eight experiments therefore require pattern reuse.
    """
    from bofire.data_models.strategies.api import DoEStrategy as DoEDataModel
    from bofire.data_models.strategies.doe import SpaceFillingCriterion
    from bofire.strategies.doe_strategy import DoEStrategy

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
    strategy = DoEStrategy(
        data_model=DoEDataModel(
            domain=domain, criterion=SpaceFillingCriterion(), seed=42
        )
    )
    before = domain.model_dump()
    candidates = strategy.ask(n_experiments)
    assert len(candidates) == n_experiments
    domain.validate_candidates(candidates, only_inputs=True, tol=1e-4)
    assert domain.model_dump() == before


def get_formula_from_string_recursion_limit():
    # save recursion limit
    recursion_limit = sys.getrecursionlimit()

    # get formula for very large model
    model = ""
    for i in range(350):
        model += f"x{i} + "
    model = model[:-3]
    model = get_formula_from_string(model_type=model)

    terms = [f"x{i}" for i in range(350)]
    terms.append("1")

    for i in range(351):
        assert np.array(model, dtype=str)[i] in terms
        assert terms[i] in model

    assert recursion_limit == sys.getrecursionlimit()


def test_get_formula_from_string():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(3)],
        outputs=[ContinuousOutput(key="y")],
    )

    # linear model
    terms = ["1", "x0", "x1", "x2"]
    model_formula = get_formula_from_string(inputs=domain.inputs, model_type="linear")
    assert all(term in terms for term in model_formula)
    assert all(term in np.array(model_formula, dtype=str) for term in terms)

    # linear and interaction
    terms = ["1", "x0", "x1", "x2", "x0:x1", "x0:x2", "x1:x2"]
    model_formula = get_formula_from_string(
        inputs=domain.inputs,
        model_type="linear-and-interactions",
    )
    assert all(term in terms for term in model_formula)
    assert all(term in np.array(model_formula, dtype=str) for term in terms)

    # linear and quadratic
    terms = ["1", "x0", "x1", "x2", "x0 ** 2", "x1 ** 2", "x2 ** 2"]
    model_formula = get_formula_from_string(
        inputs=domain.inputs,
        model_type="linear-and-quadratic",
    )
    assert all(term in terms for term in model_formula)
    assert all(term in np.array(model_formula, dtype=str) for term in terms)

    # fully quadratic
    terms = [
        "1",
        "x0",
        "x1",
        "x2",
        "x0:x1",
        "x0:x2",
        "x1:x2",
        "x0 ** 2",
        "x1 ** 2",
        "x2 ** 2",
    ]
    model_formula = get_formula_from_string(
        inputs=domain.inputs, model_type="fully-quadratic"
    )
    assert all(term in terms for term in model_formula)
    assert all(term in np.array(model_formula, dtype=str) for term in terms)

    # custom model
    terms_lhs = ["y"]
    terms_rhs = ["1", "x0", "x0 ** 2", "x0:x1"]
    model_formula = get_formula_from_string(
        inputs=domain.inputs,
        model_type="y ~ 1 + x0 + x0:x1 + {x0**2}",
        rhs_only=False,
    )
    assert all(term in terms_lhs for term in model_formula.lhs)
    assert all(term in str(model_formula.lhs) for term in terms_lhs)
    assert all(term in terms_rhs for term in model_formula.rhs)
    assert all(term in np.array(model_formula.rhs, dtype=str) for term in terms_rhs)

    # get formula without model: valid input
    model = "x1 + x2 + x3"
    model = get_formula_from_string(model_type=model)
    assert str(model) == "1 + x1 + x2 + x3"

    # get formula without model: invalid input
    with pytest.raises(AssertionError):
        model = get_formula_from_string("linear")

    # get formula for very large model
    model = ""
    for i in range(350):
        model += f"x{i} + "
    model = model[:-3]
    model = get_formula_from_string(model_type=model)

    terms = [f"x{i}" for i in range(350)]
    terms.append("1")

    for i in range(351):
        assert list(model)[i] in terms
        assert terms[i] in np.array(model, dtype=str)


def test_n_zero_eigvals_unconstrained():
    # 5 continuous
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 100)) for i in range(5)],
        outputs=[ContinuousOutput(key="y")],
    )

    assert n_zero_eigvals(domain=domain, model_type="linear") == 0
    assert n_zero_eigvals(domain=domain, model_type="linear-and-quadratic") == 0
    assert n_zero_eigvals(domain=domain, model_type="linear-and-interactions") == 0
    assert n_zero_eigvals(domain=domain, model_type="fully-quadratic") == 0


def test_n_zero_eigvals_constrained():
    # 3 continuous & 2 discrete inputs, 1 mixture constraint
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 100)),
            ContinuousInput(key="x2", bounds=(0, 100)),
            ContinuousInput(key="x3", bounds=(0, 100)),
            DiscreteInput(key="discrete1", values=[0, 1, 5]),
            DiscreteInput(key="discrete2", values=[0, 1]),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3"],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
        ],
    )

    # Comment DL: in the test in the doe package discrete was sampled as continuous
    # thus there was one degree of freedom more if quadratic terms where added.
    # Here, discretes are sampled within their respective domain, thus discrete2==discrete2**2 always
    # thus we have one degree of freedom less.
    # discrete2**2 (with only two levels) is no longer in the formula so  counts for linear-and quadratic and quadratic are updated accordingly.
    assert n_zero_eigvals(domain, "linear") == 1
    assert n_zero_eigvals(domain, "linear-and-quadratic") == 1
    assert n_zero_eigvals(domain, "linear-and-interactions") == 3
    assert n_zero_eigvals(domain, "fully-quadratic") == 6

    # TODO: NChooseK?


def test_number_of_model_terms():
    # 5 continuous inputs
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(5)],
        outputs=[ContinuousOutput(key="y")],
    )

    formula = get_formula_from_string(inputs=domain.inputs, model_type="linear")
    assert len(formula) == 6

    formula = get_formula_from_string(
        inputs=domain.inputs, model_type="linear-and-quadratic"
    )
    assert len(formula) == 11

    formula = get_formula_from_string(
        inputs=domain.inputs,
        model_type="linear-and-interactions",
    )
    assert len(formula) == 16

    formula = get_formula_from_string(
        inputs=domain.inputs, model_type="fully-quadratic"
    )
    assert len(formula) == 21

    # 3 continuous & 2 discrete inputs
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 100)),
            ContinuousInput(key="x2", bounds=(0, 100)),
            ContinuousInput(key="x3", bounds=(0, 100)),
            DiscreteInput(key="discrete1", values=[0, 1, 5]),
            DiscreteInput(key="discrete2", values=[0, 1]),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    formula = get_formula_from_string(inputs=domain.inputs, model_type="linear")
    assert len(formula) == 6

    formula = get_formula_from_string(
        inputs=domain.inputs, model_type="linear-and-quadratic"
    )
    assert len(formula) == 10  # discrete2 has only 2 levels, no quadratic term

    formula = get_formula_from_string(
        inputs=domain.inputs,
        model_type="linear-and-interactions",
    )
    assert len(formula) == 16

    formula = get_formula_from_string(
        inputs=domain.inputs, model_type="fully-quadratic"
    )
    assert len(formula) == 20  # discrete2 has only 2 levels, no quadratic term


def test_constraints_as_scipy_constraints():
    # test domains from the paper "The construction of D- and I-optimal designs for
    # mixture experiments with linear constraints on the components" by R. Coetzer and
    # L. M. Haines.
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{1}", bounds=(0, 1)),
            ContinuousInput(key=f"x{2}", bounds=(0.1, 1)),
            ContinuousInput(key=f"x{3}", bounds=(0, 0.6)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3"],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"],
                coefficients=[5, 4],
                rhs=3.9,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"],
                coefficients=[-20, 5],
                rhs=-3,
            ),
        ],
    )

    n_experiments = 2

    constraints = constraints_as_scipy_constraints(domain, n_experiments)

    for c in constraints:
        assert isinstance(c, LinearConstraint)
        assert c.A.shape == (n_experiments, len(domain.inputs) * n_experiments)
        assert len(c.lb) == n_experiments
        assert len(c.ub) == n_experiments

    A = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]) / np.sqrt(3)
    lb = np.array([1, 1]) / np.sqrt(3)
    ub = np.array([1, 1]) / np.sqrt(3)
    assert np.allclose(constraints[0].A, A)
    assert np.allclose(constraints[0].lb, lb)
    assert np.allclose(constraints[0].ub, ub)

    lb = -np.inf * np.ones(n_experiments)
    ub = 3.9 / np.linalg.norm([5, 4]) * np.ones(n_experiments)
    assert np.allclose(constraints[1].lb, lb)
    assert np.allclose(constraints[1].ub, ub)

    # domain with nonlinear constraints
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(3)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NonlinearEqualityConstraint(
                expression="x1**2 + x2**2 - 1",
                features=["x1", "x2", "x3"],
            ),
            NonlinearInequalityConstraint(
                expression="x1**2 + x2**2 - 1",
                features=["x1", "x2", "x3"],
            ),
        ],
    )

    constraints = constraints_as_scipy_constraints(domain, n_experiments)

    for c in constraints:
        assert isinstance(c, NonlinearConstraint)
        assert len(c.lb) == n_experiments
        assert len(c.ub) == n_experiments
        assert np.allclose(c.fun(np.array([1, 1, 1, 1, 1, 1])), [1, 1])

    # TODO NChooseKConstraint requires input lower_bounds to be 0.
    # can we lift this requirement?

    inputs = [ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(4)]

    domain = Domain.from_lists(
        inputs=inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=[i.key for i in inputs],
                max_count=2,
                min_count=0,
                none_also_valid=True,
            ),
        ],
    )
    n_experiments = 1

    constraints = constraints_as_scipy_constraints(
        domain,
        n_experiments,
        ignore_nchoosek=True,
    )
    assert len(constraints) == 0

    constraints = constraints_as_scipy_constraints(
        domain,
        n_experiments,
        ignore_nchoosek=False,
    )
    assert len(constraints) == 1
    assert isinstance(constraints[0], NonlinearConstraint)
    assert np.allclose(
        constraints[0].fun(np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])),
        [2, 0, 0],
    )

    # domain with batch constraint
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(3)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            InterpointEqualityConstraint(features=["x0"], multiplicity=3),
        ],
    )
    n_experiments = 5

    constraints = constraints_as_scipy_constraints(domain, n_experiments)
    assert len(constraints) == 1
    assert isinstance(constraints[0], LinearConstraint)
    A = np.array(
        [
            [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0],
        ],
        dtype=float,
    )
    assert np.allclose(constraints[0].A, A)
    assert np.allclose(constraints[0].lb, np.zeros(3))
    assert np.allclose(constraints[0].ub, np.zeros(3))


def test_ConstraintWrapper():
    # define domain with all types of constraints
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(4)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3", "x4"],
                coefficients=[1, 1, 1, 1],
                rhs=1,
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "x3", "x4"],
                coefficients=[1, 1, 1, 1],
                rhs=1,
            ),
            NonlinearEqualityConstraint(
                expression="x1**2 + x2**2 + x3**2 + x4**2 - 1",
                features=["x1", "x2", "x3", "x4"],
                jacobian_expression="[2*x1, 2*x2, 2*x3, 2*x4]",
            ),
            NonlinearInequalityConstraint(
                expression="x1**2 + x2**2 + x3**2 + x4**2 - 1",
                features=["x1", "x2", "x3", "x4"],
                jacobian_expression="[2*x1, 2*x2, 2*x3, 2*x4]",
            ),
            NonlinearEqualityConstraint(
                expression="x1**2 + x4**2 - 1",
                features=["x1", "x4"],
                jacobian_expression="[2*x1, 2*x4]",
            ),
        ],
    )

    x = np.array([[1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], [3, 2, 1, 0]]).flatten()

    # linear equality
    c = ConstraintWrapper(domain.constraints[0], domain, n_experiments=3)
    assert np.allclose(c(x), np.array([1.5, 0.5, 2.5]))
    assert np.allclose(
        c.jacobian(x),
        0.5
        * np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            ],
        ),
    )

    # linear inequality
    c = ConstraintWrapper(domain.constraints[1], domain, n_experiments=3)
    assert np.allclose(c(x), np.array([1.5, 0.5, 2.5]))
    assert np.allclose(
        c.jacobian(x),
        0.5
        * np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            ],
        ),
    )

    # nonlinear equality
    c = ConstraintWrapper(domain.constraints[2], domain, n_experiments=3)
    assert np.allclose(c(x), np.array([3, 0, 13]))
    assert np.allclose(
        c.jacobian(x),
        np.array(
            [
                [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 2, 0],
            ],
        ),
    )

    # nonlinear inequality
    c = ConstraintWrapper(domain.constraints[3], domain, n_experiments=3)
    assert np.allclose(c(x), np.array([3, 0, 13]))
    assert np.allclose(
        c.jacobian(x),
        np.array(
            [
                [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 2, 0],
            ],
        ),
    )

    # constraint not containing all inputs from domain
    c = ConstraintWrapper(domain.constraints[4], domain, n_experiments=3)
    assert np.allclose(c(x), np.array([1, -0.5, 8]))
    assert np.allclose(
        c.jacobian(x),
        np.array(
            [
                [2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0],
            ],
        ),
    )
    assert np.allclose(
        c.jacobian(x, sparse=True), np.array([2, 0, 0, 2, 1, 0, 0, 1, 6, 0, 0, 0])
    )

    assert np.allclose(
        c.hessian(x),
        [
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        ],
    )


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="cyipopt required")
def test_minimize():
    # Run _minimize() with and without ipopt
    n_experiments = 4
    criterion = DOptimalityCriterion(formula="linear")
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3"],
                coefficients=[1, 1, 1],
                rhs=1,
            ),
        ],
    )

    objective_function = get_objective_function(
        criterion, domain=domain, n_experiments=n_experiments
    )

    x0 = (
        domain.inputs.sample(n=n_experiments, method=SamplingMethodEnum.UNIFORM)
        .to_numpy()
        .flatten()
    )
    constraints = constraints_as_scipy_constraints(
        domain,
        n_experiments,
        ignore_nchoosek=True,
    )
    bounds = nchoosek_constraints_as_bounds(domain, n_experiments)

    result_ipopt = _minimize(
        objective_function=objective_function,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        ipopt_options={"max_iter": 500, "print_level": 0},
        use_hessian=False,
        use_cyipopt=True,
    )

    result_scipy = _minimize(
        objective_function=objective_function,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        ipopt_options={"max_iter": 500},
        use_hessian=False,
        use_cyipopt=False,
    )

    for i in range(n_experiments):
        assert np.any(
            [
                np.allclose(result_ipopt[j], result_scipy[i])
                for j in range(n_experiments)
            ]
        )
        assert np.any(
            [
                np.allclose(result_scipy[j], result_ipopt[i])
                for j in range(n_experiments)
            ]
        )


def test_nchoosek_constraints_as_bounds():
    # define domain: no NChooseK constraints
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i + 1}",
                bounds=(0, 1),
            )
            for i in range(5)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    bounds = nchoosek_constraints_as_bounds(domain, n_experiments=4)
    assert len(bounds) == 20
    _bounds = [
        (p.lower_bound, p.upper_bound)
        for p in domain.inputs
        if isinstance(p, ContinuousInput)
    ] * 4
    for i in range(20):
        assert _bounds[i] == bounds[i]


def test_convert_formula_to_string():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(3)],
        outputs=[ContinuousOutput(key="y")],
    )

    formula = get_formula_from_string(
        inputs=domain.inputs, model_type="fully-quadratic"
    )

    formula_str = convert_formula_to_string(domain=domain, formula=formula)
    assert (
        formula_str
        == "torch.vstack([torch.ones_like(x0), x0, x1, x2, x0 ** 2, x1 ** 2, x2 ** 2,"
        + " x0*x1, x0*x2, x1*x2, ]).T"
    )


def test_formula_discrete_handled_like_continuous():
    domain_w_discrete = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=[0, 1]) for i in range(3)]
        + [DiscreteInput(key=f"x{i}", values=[0, 1, 2]) for i in range(3, 5)],
        outputs=[ContinuousOutput(key="y")],
    )
    domain_wo_discrete = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=[0, 1]) for i in range(3)]
        + [ContinuousInput(key=f"x{i}", bounds=[0, 1]) for i in range(3, 5)],
        outputs=[ContinuousOutput(key="y")],
    )

    for model_type in [
        "linear",
        "linear-and-interactions",
        "linear-and-quadratic",
        "fully-quadratic",
    ]:
        formula_w_discrete = get_formula_from_string(
            inputs=domain_w_discrete.inputs, model_type=model_type
        )
        formula_wo_discrete = get_formula_from_string(
            inputs=domain_wo_discrete.inputs, model_type=model_type
        )
        assert formula_w_discrete == formula_wo_discrete


def test_formula_discrete_too_few_levels():
    domain_w_discrete = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=[0, 1]) for i in range(3)]
        + [DiscreteInput(key=f"x{i}", values=[0, 1]) for i in range(3, 5)],
        outputs=[ContinuousOutput(key="y")],
    )
    domain_wo_discrete = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=[0, 1]) for i in range(3)]
        + [ContinuousInput(key=f"x{i}", bounds=[0, 1]) for i in range(3, 5)],
        outputs=[ContinuousOutput(key="y")],
    )

    for model_type in [
        "linear",
        "linear-and-interactions",
    ]:
        formula_w_discrete = get_formula_from_string(
            inputs=domain_w_discrete.inputs, model_type=model_type
        )
        formula_wo_discrete = get_formula_from_string(
            inputs=domain_wo_discrete.inputs, model_type=model_type
        )
        assert formula_w_discrete == formula_wo_discrete

    for model_type in [
        "linear-and-quadratic",
        "fully-quadratic",
    ]:
        formula_w_discrete = str(
            get_formula_from_string(
                inputs=domain_w_discrete.inputs, model_type=model_type
            )
        )

        formula_wo_discrete = str(
            get_formula_from_string(
                inputs=domain_wo_discrete.inputs, model_type=model_type
            )
        ).replace(" + x3 ** 2 + x4 ** 2", "")
        assert formula_w_discrete == formula_wo_discrete


def test_formula_str_to_fully_continuous():
    # Create a small example problem with categorical, continuous, and discrete variables
    inputs = Inputs(
        features=[
            CategoricalInput(
                key="color",
                categories=["red", "blue", "green"],
            ),
            ContinuousInput(
                key="color_intensity",
                bounds=(0.0, 1.0),
            ),
            CategoricalInput(
                key="material",
                categories=["plastic", "metal"],
            ),
            ContinuousInput(
                key="temperature",
                bounds=(20.0, 100.0),
            ),
            DiscreteInput(
                key="pressure",
                values=[1.0, 2.0, 3.0, 5.0, 10.0],
            ),
        ]
    )

    # Define a custom formula with interactions among categorical variables
    # This includes interaction between color and material
    custom_formula = (
        "color + material + temperature + pressure + color:material + color_intensity"
    )

    # Convert to fully continuous representation
    continuous_formula = formula_str_to_fully_continuous(
        formula_str=custom_formula,
        inputs=inputs,
    )

    # Assert the expected formula explicitly
    expected_formula = "1 + aux_color_red + aux_color_blue + aux_material_plastic + temperature + pressure + color_intensity + aux_color_red:aux_material_plastic + aux_color_blue:aux_material_plastic"
    assert (
        str(continuous_formula) == expected_formula
    ), f"Expected: {expected_formula}\nGot: {continuous_formula}"

    custom_formula = "color + temperature + pressure + color_intensity"
    continuous_formula = formula_str_to_fully_continuous(
        formula_str=custom_formula,
        inputs=inputs,
    )
    expected_formula = (
        "1 + aux_color_red + aux_color_blue + temperature + pressure + color_intensity"
    )
    assert (
        str(continuous_formula) == expected_formula
    ), f"Expected: {expected_formula}\nGot: {continuous_formula}"

    custom_formula = "material + temperature + pressure"
    continuous_formula = formula_str_to_fully_continuous(
        formula_str=custom_formula,
        inputs=inputs,
    )
    expected_formula = "1 + aux_material_plastic + temperature + pressure"
    assert (
        str(continuous_formula) == expected_formula
    ), f"Expected: {expected_formula}\nGot: {continuous_formula}"

    custom_formula = "temperature:material + color_intensity + pressure:color"

    continuous_formula = formula_str_to_fully_continuous(
        formula_str=custom_formula,
        inputs=inputs,
    )
    expected_formula = "1 + color_intensity + temperature:aux_material_plastic + pressure:aux_color_red + pressure:aux_color_blue"
    assert (
        str(continuous_formula) == expected_formula
    ), f"Expected: {expected_formula}\nGot: {continuous_formula}"

    custom_formula = "1 + color + material + temperature + pressure + color:material + temperature:material + pressure:color + color_intensity"
    continuous_formula = formula_str_to_fully_continuous(
        formula_str=custom_formula,
        inputs=inputs,
    )
    expected_formula = "1 + aux_color_red + aux_color_blue + aux_material_plastic + temperature + pressure + color_intensity + aux_color_red:aux_material_plastic + aux_color_blue:aux_material_plastic + temperature:aux_material_plastic + pressure:aux_color_red + pressure:aux_color_blue"
    assert (
        str(continuous_formula) == expected_formula
    ), f"Expected: {expected_formula}\nGot: {continuous_formula}"


def test_formula_str_does_not_match_discrete_levels_emmits_warning():
    # Create a small example problem with categorical, continuous, and discrete variables

    inputs = Inputs(
        features=[
            CategoricalInput(
                key="color",
                categories=["red", "blue", "green"],
            ),
            ContinuousInput(
                key="color_intensity",
                bounds=(0.0, 1.0),
            ),
            CategoricalInput(
                key="material",
                categories=["plastic", "metal"],
            ),
            ContinuousInput(
                key="temperature",
                bounds=(20.0, 100.0),
            ),
            DiscreteInput(
                key="pressure",
                values=[0, 1],
            ),
        ]
    )

    # Define a custom formula with interactions among categorical variables
    # This includes interaction between color and material
    custom_formula = "color + material + temperature + { pressure ** 2 } + color:material + color_intensity"
    with pytest.warns(
        UserWarning,
        match="Discrete input pressure with 2 levels cannot represent a term of order 2 or higher.",
    ):
        formula_str_to_fully_continuous(
            formula_str=custom_formula,
            inputs=inputs,
        )


def test_formula_str_to_fully_continuous_only_categoricals():
    # Create a small example problem with only categorical variables
    inputs = Inputs(
        features=[
            CategoricalInput(
                key="color",
                categories=["red", "blue", "green"],
            ),
            CategoricalInput(
                key="material",
                categories=["plastic", "metal"],
            ),
            CategoricalInput(
                key="material_shape",
                categories=["circle", "square"],
            ),
        ]
    )

    # Define a custom formula with interactions among categorical variables
    custom_formula = "color + material + color:material + material:material_shape"
    # Convert to fully continuous representation
    continuous_formula = formula_str_to_fully_continuous(
        formula_str=custom_formula,
        inputs=inputs,
    )
    # Assert the expected formula explicitly
    expected_formula = (
        "1 + aux_color_red + aux_color_blue + aux_material_plastic"
        + " + aux_color_red:aux_material_plastic + aux_color_blue:aux_material_plastic"
        + " + aux_material_plastic:aux_material_shape_circle"
    )
    assert (
        str(continuous_formula) == expected_formula
    ), f"Expected: {expected_formula}\nGot: {continuous_formula}"

    custom_formula = "material:color + material_shape"
    continuous_formula = formula_str_to_fully_continuous(
        formula_str=custom_formula,
        inputs=inputs,
    )
    expected_formula = "1 + aux_material_shape_circle + aux_material_plastic:aux_color_red + aux_material_plastic:aux_color_blue"
    assert (
        str(continuous_formula) == expected_formula
    ), f"Expected: {expected_formula}\nGot: {continuous_formula}"


def only_continuous_inputs_formula_str_to_fully_continuous():
    # Create a small example problem with only continuous and discrete variables
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="length",
                bounds=(0.0, 10.0),
            ),
            DiscreteInput(
                key="width",
                values=[1.0, 2.0, 3.0],
            ),
            ContinuousInput(
                key="height",
                bounds=(5.0, 15.0),
            ),
        ]
    )

    # Define a custom formula
    custom_formula = "length + width + height + length:height"

    # Convert to fully continuous representation
    continuous_formula = formula_str_to_fully_continuous(
        formula_str=custom_formula,
        inputs=inputs,
    )

    # Assert the expected formula explicitly
    expected_formula = "1 + length + width + height + length:height"
    assert (
        str(continuous_formula) == expected_formula
    ), f"Expected: {expected_formula}\nGot: {continuous_formula}"


def test_nchoosek_bounds_known_patterns():
    """Test nchoosek_constraints_as_bounds against known expected activity patterns.

    For 3 features with min_count=1, max_count=2, the complete set of deactivation
    patterns is deterministic (only the order within experiments is random):
      k=1 (2 inactive): (0,0,1), (0,1,0), (1,0,0)
      k=2 (1 inactive): (0,1,1), (1,0,1), (1,1,0)
    With n_experiments >= 6, every pattern must appear at least once.
    """
    n_features = 3
    d = Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{i}", bounds=(0.0, 1.0)) for i in range(n_features)
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=["x0", "x1", "x2"],
                min_count=1,
                max_count=2,
                none_also_valid=False,
            )
        ],
    )
    n_experiments = 12
    bounds = nchoosek_constraints_as_bounds(d, n_experiments=n_experiments)

    D = n_features
    assert len(bounds) == D * n_experiments

    # extract the activity pattern (1=active, 0=pinned-to-zero) per experiment
    observed_patterns = set()
    for i in range(n_experiments):
        exp_bounds = bounds[i * D : (i + 1) * D]
        pattern = tuple(1 if b != (0.0, 0.0) else 0 for b in exp_bounds)
        observed_patterns.add(pattern)
        # every active slot must keep its original bounds
        for j, b in enumerate(exp_bounds):
            if b != (0.0, 0.0):
                assert b == (
                    0.0,
                    1.0,
                ), f"exp {i}, feature {j}: expected (0.0, 1.0), got {b}"

    # the expected set of all patterns for min_count=1, max_count=2, 3 features
    expected_patterns = {
        # k=1: exactly 1 active feature
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        # k=2: exactly 2 active features
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    }
    assert observed_patterns == expected_patterns, (
        f"Expected patterns {sorted(expected_patterns)}, "
        f"got {sorted(observed_patterns)}"
    )

    # every pattern must have between min_count and max_count active features
    for pat in observed_patterns:
        active = sum(pat)
        assert 1 <= active <= 2, f"Pattern {pat} has {active} active features"


def test_nchoosek_bounds_none_also_valid():
    """Test none_also_valid behavior for the all-zero pattern."""

    n_features = 3

    # --- Case 1: none_also_valid=False with min_count=0 ---
    d_no_none = Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{i}", bounds=(1.0, 2.0), allow_zero=True)
            for i in range(n_features)
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=["x0", "x1", "x2"],
                min_count=0,
                max_count=2,
                none_also_valid=False,
            )
        ],
    )

    n_experiments = 12
    bounds = nchoosek_constraints_as_bounds(d_no_none, n_experiments=n_experiments)

    D = n_features
    observed_patterns = set()
    for i in range(n_experiments):
        exp_bounds = bounds[i * D : (i + 1) * D]
        pattern = tuple(1 if b != (0.0, 0.0) else 0 for b in exp_bounds)
        observed_patterns.add(pattern)

    # none_also_valid=False: all-zero pattern should NOT appear
    expected_patterns = {
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
    }
    assert observed_patterns == expected_patterns, (
        f"Expected patterns {sorted(expected_patterns)}, "
        f"got {sorted(observed_patterns)}"
    )

    # --- Case 2: none_also_valid=True with min_count=0 ---
    # When min_count=0, the all-zero pattern is NOT added in bounds
    # (it is handled at validation level by is_fulfilled / domain.py).
    d_with_none = Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{i}", bounds=(1.0, 2.0), allow_zero=True)
            for i in range(n_features)
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=["x0", "x1", "x2"],
                min_count=0,
                max_count=2,
                none_also_valid=True,
            )
        ],
    )
    bounds = nchoosek_constraints_as_bounds(d_with_none, n_experiments=n_experiments)

    observed_patterns = set()
    for i in range(n_experiments):
        exp_bounds = bounds[i * D : (i + 1) * D]
        pattern = tuple(1 if b != (0.0, 0.0) else 0 for b in exp_bounds)
        observed_patterns.add(pattern)

    # Same patterns as Case 1: all-zero is NOT added in bounds when min_count=0
    assert observed_patterns == expected_patterns, (
        f"Expected patterns {sorted(expected_patterns)}, "
        f"got {sorted(observed_patterns)}"
    )

    # --- Case 3: none_also_valid=True with min_count > 0 ---
    # none_also_valid does NOT affect bounds generation (only is_fulfilled
    # and domain.py enumeration).  With min_count=2, max_count=2, we only
    # get the C(3,1) = 3 patterns with exactly 2 active features.
    d_min_gt_zero = Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{i}", bounds=(1.0, 2.0), allow_zero=True)
            for i in range(n_features)
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=["x0", "x1", "x2"],
                min_count=2,
                max_count=2,
                none_also_valid=True,
            )
        ],
    )
    bounds = nchoosek_constraints_as_bounds(d_min_gt_zero, n_experiments=n_experiments)

    observed_patterns = set()
    for i in range(n_experiments):
        exp_bounds = bounds[i * D : (i + 1) * D]
        pattern = tuple(1 if b != (0.0, 0.0) else 0 for b in exp_bounds)
        observed_patterns.add(pattern)

    expected_patterns_min_gt_zero = {
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    }
    assert observed_patterns == expected_patterns_min_gt_zero, (
        f"Expected patterns {sorted(expected_patterns_min_gt_zero)}, "
        f"got {sorted(observed_patterns)}"
    )


def test_multi_nchoosek_bounds_known_patterns():
    """Test nchoosek_constraints_as_bounds against known expected activity patterns."""
    n_features = 3
    d = Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{i}", bounds=(0.0, 1.0)) for i in range(n_features)
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=["x0", "x1"],
                min_count=1,
                max_count=1,
                none_also_valid=False,
            ),
            NChooseKConstraint(
                features=["x1", "x2"],
                min_count=1,
                max_count=1,
                none_also_valid=False,
            ),
        ],
    )
    n_experiments = 12
    bounds = nchoosek_constraints_as_bounds(d, n_experiments=n_experiments)

    D = n_features
    assert len(bounds) == D * n_experiments

    # extract the activity pattern (1=active, 0=pinned-to-zero) per experiment
    observed_patterns = set()
    for i in range(n_experiments):
        exp_bounds = bounds[i * D : (i + 1) * D]
        pattern = tuple(1 if b != (0.0, 0.0) else 0 for b in exp_bounds)
        observed_patterns.add(pattern)
        # every active slot must keep its original bounds
        for j, b in enumerate(exp_bounds):
            if b != (0.0, 0.0):
                assert b == (
                    0.0,
                    1.0,
                ), f"exp {i}, feature {j}: expected (0.0, 1.0), got {b}"

    expected_patterns = {
        (0, 1, 0),  # x1 active, x0 and x2 inactive
        (1, 0, 1),  # x0 and x2 active, x1 inactive
    }
    assert observed_patterns == expected_patterns, (
        f"Expected patterns {sorted(expected_patterns)}, "
        f"got {sorted(observed_patterns)}"
    )

    # every pattern must have between min_count and max_count active features
    for pat in observed_patterns:
        active = sum(pat)
        assert 1 <= active <= 2, f"Pattern {pat} has {active} active features"


if __name__ == "__main__":
    test_multi_nchoosek_bounds_known_patterns()
