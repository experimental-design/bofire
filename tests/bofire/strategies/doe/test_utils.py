import sys

import numpy as np
import pytest
from scipy.optimize import LinearConstraint
from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain import Domain
from bofire.domain.features import ContinuousInput, ContinuousOutput, DiscreteInput
from bofire.strategies.doe.utils import (
    ConstraintWrapper,
    a_optimality,
    check_nchoosek_constraints_as_bounds,
    constraints_as_scipy_constraints,
    d_optimality,
    g_efficiency,
    get_formula_from_string,
    metrics,
    n_zero_eigvals,
    nchoosek_constraints_as_bounds,
)


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
        assert model.terms[i] in terms
        assert terms[i] in model.terms

    assert recursion_limit == sys.getrecursionlimit()


def test_get_formula_from_string():
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i}", lower_bound=0, upper_bound=1) for i in range(3)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    # linear model
    terms = ["1", "x0", "x1", "x2"]
    model_formula = get_formula_from_string(domain=domain, model_type="linear")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and interaction
    terms = ["1", "x0", "x1", "x2", "x0:x1", "x0:x2", "x1:x2"]
    model_formula = get_formula_from_string(
        domain=domain, model_type="linear-and-interactions"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # linear and quadratic
    terms = ["1", "x0", "x1", "x2", "x0**2", "x1**2", "x2**2"]
    model_formula = get_formula_from_string(
        domain=domain, model_type="linear-and-quadratic"
    )
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # fully quadratic
    terms = [
        "1",
        "x0",
        "x1",
        "x2",
        "x0:x1",
        "x0:x2",
        "x1:x2",
        "x0**2",
        "x1**2",
        "x2**2",
    ]
    model_formula = get_formula_from_string(domain=domain, model_type="fully-quadratic")
    assert all(term in terms for term in model_formula.terms)
    assert all(term in model_formula.terms for term in terms)

    # custom model
    terms_lhs = ["y"]
    terms_rhs = ["1", "x0", "x0**2", "x0:x1"]
    model_formula = get_formula_from_string(
        domain=domain,
        model_type="y ~ 1 + x0 + x0:x1 + {x0**2}",
        rhs_only=False,
    )
    assert all(term in terms_lhs for term in model_formula.terms.lhs)
    assert all(term in model_formula.terms.lhs for term in terms_lhs)
    assert all(term in terms_rhs for term in model_formula.terms.rhs)
    assert all(term in model_formula.terms.rhs for term in terms_rhs)

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
        assert model.terms[i] in terms
        assert terms[i] in model.terms


def test_n_zero_eigvals_unconstrained():
    # 5 continous
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=100)
            for i in range(5)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    assert n_zero_eigvals(domain=domain, model_type="linear") == 0
    assert n_zero_eigvals(domain=domain, model_type="linear-and-quadratic") == 0
    assert n_zero_eigvals(domain=domain, model_type="linear-and-interactions") == 0
    assert n_zero_eigvals(domain=domain, model_type="fully-quadratic") == 0


def test_n_zero_eigvals_constrained():
    # 3 continuous & 2 discrete input_features, 1 mixture constraint
    domain = Domain(
        input_features=[
            ContinuousInput(key="x1", lower_bound=0, upper_bound=100),
            ContinuousInput(key="x2", lower_bound=0, upper_bound=100),
            ContinuousInput(key="x3", lower_bound=0, upper_bound=100),
            DiscreteInput(key="discrete1", values=[0, 1, 5]),
            DiscreteInput(key="discrete2", values=[0, 1]),
        ],
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3"], coefficients=[1, 1, 1], rhs=1
            )
        ],
    )

    # Comment DL: in the test in the doe package discrete was sampled as continuous
    # thus there was one degree of freedom more if quadratic terms where added.
    # Here, discretes are sampled within their respective domain, thus discrete2==discrete2**2 always
    # thus we have one degree of freedom less.
    assert n_zero_eigvals(domain, "linear") == 1
    assert n_zero_eigvals(domain, "linear-and-quadratic") == 2
    assert n_zero_eigvals(domain, "linear-and-interactions") == 3
    assert n_zero_eigvals(domain, "fully-quadratic") == 7

    # TODO: NChooseK?


def test_number_of_model_terms():
    # 5 continous input_features
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i}", lower_bound=0, upper_bound=1) for i in range(5)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    formula = get_formula_from_string(domain=domain, model_type="linear")
    assert len(formula.terms) == 6

    formula = get_formula_from_string(domain=domain, model_type="linear-and-quadratic")
    assert len(formula.terms) == 11

    formula = get_formula_from_string(
        domain=domain, model_type="linear-and-interactions"
    )
    assert len(formula.terms) == 16

    formula = get_formula_from_string(domain=domain, model_type="fully-quadratic")
    assert len(formula.terms) == 21

    # 3 continuous & 2 discrete input_features
    domain = Domain(
        input_features=[
            ContinuousInput(key="x1", lower_bound=0, upper_bound=100),
            ContinuousInput(key="x2", lower_bound=0, upper_bound=100),
            ContinuousInput(key="x3", lower_bound=0, upper_bound=100),
            DiscreteInput(key="discrete1", values=[0, 1, 5]),
            DiscreteInput(key="discrete2", values=[0, 1]),
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    formula = get_formula_from_string(domain=domain, model_type="linear")
    assert len(formula.terms) == 6

    formula = get_formula_from_string(domain=domain, model_type="linear-and-quadratic")
    assert len(formula.terms) == 11

    formula = get_formula_from_string(
        domain=domain, model_type="linear-and-interactions"
    )
    assert len(formula.terms) == 16

    formula = get_formula_from_string(domain=domain, model_type="fully-quadratic")
    assert len(formula.terms) == 21


def test_constraints_as_scipy_constraints():
    # test domains from the paper "The construction of D- and I-optimal designs for
    # mixture experiments with linear constraints on the components" by R. Coetzer and
    # L. M. Haines.

    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
            ContinuousInput(key=f"x{2}", lower_bound=0.1, upper_bound=1),
            ContinuousInput(key=f"x{3}", lower_bound=0, upper_bound=0.6),
        ],
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3"], coefficients=[1, 1, 1], rhs=1
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"], coefficients=[5, 4], rhs=3.9
            ),
            LinearInequalityConstraint(
                features=["x1", "x2"], coefficients=[-20, 5], rhs=-3
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
    lb = np.array([1, 1]) / np.sqrt(3) - 0.001
    ub = np.array([1, 1]) / np.sqrt(3) + 0.001
    assert np.allclose(constraints[0].A, A)
    assert np.allclose(constraints[0].lb, lb)
    assert np.allclose(constraints[0].ub, ub)

    # lb = -np.inf * np.ones(n_experiments)
    # ub = -0.1 * np.ones(n_experiments)
    # assert np.allclose(constraints[1].lb, lb)
    # assert np.allclose(constraints[1].ub, ub)

    # TODO? Reimplement nonlinear constrains?
    # where they ever funtional?

    # domain with nonlinear constraints
    # domain = Domain(
    #     input_features=[
    #         ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
    #         for i in range(3)
    #     ],
    #     output_features=[ContinuousOutput(key="y")],
    #     constraints=[
    #         opti.NonlinearEquality("x1**2 + x2**2 - 1"),
    #         opti.NonlinearInequality("x1**2 + x2**2 - 1"),
    #     ],
    # )

    # constraints = constraints_as_scipy_constraints(domain, n_experiments)

    # for c in constraints:
    #     assert isinstance(c, NonlinearConstraint)
    #     assert len(c.lb) == n_experiments
    #     assert len(c.ub) == n_experiments
    #     assert np.allclose(c.fun(np.array([1, 1, 1, 1, 1, 1])), [1, 1])

    # TODO NChooseKConstraint requires input lower_bounds to be 0.
    # can we lift this requirment?

    input_features = [
        ContinuousInput(key=f"x{i}", lower_bound=0, upper_bound=1) for i in range(4)
    ]

    domain = Domain(
        input_features=input_features,
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            NChooseKConstraint(
                features=[i.key for i in input_features],
                max_count=2,
                min_count=0,
                none_also_valid=True,
            )
        ],
    )
    n_experiments = 1

    constraints = constraints_as_scipy_constraints(domain, n_experiments)
    assert len(constraints) == 0


def test_ConstraintWrapper():
    # define domain with all types of constraints
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3", "x4"], coefficients=[1, 1, 1, 1], rhs=1
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "x3", "x4"], coefficients=[1, 1, 1, 1], rhs=1
            ),
            # TODO? Reimplement nonlinear constrains?
            # where they ever funtional?
            # opti.NonlinearEquality("x1**2 + x2**2 + x3**2 + x4**2 - 1"),
            # opti.NonlinearInequality("x1**2 + x2**2 + x3**2 + x4**2 - 1"),
            NChooseKConstraint(
                features=["x1", "x2", "x3", "x4"],
                max_count=3,
                min_count=0,
                none_also_valid=True,
            ),
        ],
    )

    x = np.array([[1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], [3, 2, 1, 0]]).flatten()

    # linear equality
    c = ConstraintWrapper(domain.constraints[0], domain, tol=0)
    assert np.allclose(c(x), np.array([1.5, 0.5, 2.5]))

    # linear inequaity
    c = ConstraintWrapper(domain.constraints[1], domain, tol=0)
    assert np.allclose(c(x), np.array([1.5, 0.5, 2.5]))

    # # nonlinear equality
    # c = ConstraintWrapper(domain.constraints[2], domain, tol=0)
    # assert np.allclose(c(x), np.array([3, 0, 13]))

    # # nonlinear inequality
    # c = ConstraintWrapper(domain.constraints[3], domain, tol=0)
    # assert np.allclose(c(x), np.array([3, 0, 13]))

    # nchoosek constraint
    with pytest.raises(NotImplementedError):
        c = ConstraintWrapper(domain.constraints[2], domain, tol=0)
        assert np.allclose(c(x), np.array([1, 0.5, 0]))


def test_d_optimality():
    # define model matrix: full rank
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )
    assert np.allclose(d_optimality(X), np.linalg.slogdet(X.T @ X)[1])

    # define model matrix: not full rank
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1 / 3, 1 / 3, 1 / 3],
        ]
    )
    assert np.allclose(d_optimality(X), np.sum(np.log(np.linalg.eigvalsh(X.T @ X)[1:])))


def test_a_optimality():
    # define model matrix: full rank
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )
    assert np.allclose(a_optimality(X), np.sum(1 / (np.linalg.eigvalsh(X.T @ X))))

    # define model matrix: not full rank
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 1 / 3, 1 / 3, 1 / 3],
        ]
    )
    assert np.allclose(a_optimality(X), np.sum(1 / (np.linalg.eigvalsh(X.T @ X)[1:])))


def test_g_efficiency():
    # define model matrix and domain: no constraints
    X = np.array(
        [
            [1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1],
        ]
    )

    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0.95, upper_bound=1.0)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
    )
    assert np.allclose(g_efficiency(X, domain), 0.333, atol=5e-3)

    # define domain: sampling not implemented
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1.0)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3", "x4"], coefficients=[1, 1, 1, 1], rhs=1
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "x3", "x4"],
                coefficients=[-1, -1, -1, -1],
                rhs=-0.95,
            ),
            NChooseKConstraint(
                features=["x1", "x2", "x3", "x4"],
                max_count=1,
                min_count=0,
                none_also_valid=True,
            ),
        ],
    )
    with pytest.raises(Exception):
        g_efficiency(X, domain, n_samples=1)


def test_metrics():
    # define model matrix
    X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )

    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0.95, upper_bound=1.0)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    d = metrics(X, domain)
    assert d.index[0] == "D-optimality"
    assert d.index[1] == "A-optimality"
    assert d.index[2] == "G-efficiency"
    assert np.allclose(
        d,
        np.array([d_optimality(X), a_optimality(X), g_efficiency(X, domain)]),
        rtol=0.05,
    )

    # define domain: sampling not implemented
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1.0)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(
                features=["x1", "x2", "x3", "x4"], coefficients=[1, 1, 1, 1], rhs=1
            ),
            LinearInequalityConstraint(
                features=["x1", "x2", "x3", "x4"],
                coefficients=[-1, -1, -1, -1],
                rhs=-0.95,
            ),
            NChooseKConstraint(
                features=["x1", "x2", "x3", "x4"],
                max_count=1,
                min_count=0,
                none_also_valid=True,
            ),
        ],
    )
    with pytest.warns(UserWarning):
        d = metrics(X, domain, n_samples=1)
    assert d.index[0] == "D-optimality"
    assert d.index[1] == "A-optimality"
    assert d.index[2] == "G-efficiency"
    assert np.allclose(d, np.array([d_optimality(X), a_optimality(X), 0]))


def test_check_nchoosek_constraints_as_bounds():
    # define domain: possible to formulate as bounds, no NChooseK constraints
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
    )
    check_nchoosek_constraints_as_bounds(domain)

    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=-1, upper_bound=1)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
        constraints=[],
    )
    check_nchoosek_constraints_as_bounds(domain)

    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=-np.inf, upper_bound=1)
            for i in range(4)
        ],
        output_features=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(features=["x1", "x2"], coefficients=[1, 1], rhs=0)
        ],
    )
    check_nchoosek_constraints_as_bounds(domain)

    # define domain: possible to formulate as bounds, with NChooseK and other constraints
    with pytest.raises(ValueError):
        domain = Domain(
            input_features=[
                ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
                ContinuousInput(key=f"x{2}", lower_bound=-1, upper_bound=1),
                ContinuousInput(key=f"x{3}", lower_bound=-2, upper_bound=1),
                ContinuousInput(key=f"x{4}", lower_bound=-3, upper_bound=1),
            ],
            output_features=[ContinuousOutput(key="y")],
            constraints=[
                LinearEqualityConstraint(
                    features=["x1", "x2"], coefficients=[1, 1], rhs=0
                ),
                LinearInequalityConstraint(
                    features=["x3", "x4"], coefficients=[1, 1], rhs=0
                ),
                NChooseKConstraint(
                    features=["x1", "x2"],
                    max_count=1,
                    min_count=0,
                    none_also_valid=True,
                ),
                NChooseKConstraint(
                    features=["x3", "x4"],
                    max_count=1,
                    min_count=0,
                    none_also_valid=True,
                ),
            ],
        )
        check_nchoosek_constraints_as_bounds(domain)

    # define domain: not possible to formulate as bounds, invalid bounds
    with pytest.raises(ValueError):
        domain = Domain(
            input_features=[
                ContinuousInput(key=f"x{i+1}", lower_bound=-1, upper_bound=-0.1)
                for i in range(4)
            ],
            output_features=[ContinuousOutput(key="y")],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    max_count=1,
                    min_count=0,
                    none_also_valid=True,
                ),
            ],
        )
        with pytest.raises(ValueError):
            check_nchoosek_constraints_as_bounds(domain)

    with pytest.raises(ValueError):
        domain = Domain(
            input_features=[
                ContinuousInput(key=f"x{1}", lower_bound=0, upper_bound=1),
                ContinuousInput(key=f"x{2}", lower_bound=0, upper_bound=1),
                ContinuousInput(key=f"x{3}", lower_bound=0, upper_bound=1),
                ContinuousInput(key=f"x{4}", lower_bound=0, upper_bound=1),
            ],
            output_features=[ContinuousOutput(key="y")],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    max_count=1,
                    min_count=0,
                    none_also_valid=True,
                ),
                NChooseKConstraint(
                    features=["x1", "x2"],
                    max_count=2,
                    min_count=0,
                    none_also_valid=True,
                ),
            ],
        )
        with pytest.raises(ValueError):
            check_nchoosek_constraints_as_bounds(domain)

    # define domain: not possible to formulate as bounds, names parameters of two NChooseK overlap
    with pytest.raises(ValueError):
        domain = Domain(
            input_features=[
                ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
                for i in range(4)
            ],
            output_features=[ContinuousOutput(key="y")],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    max_count=1,
                    min_count=0,
                    none_also_valid=True,
                ),
                NChooseKConstraint(
                    features=["x2", "x3", "x4"],
                    max_count=2,
                    min_count=0,
                    none_also_valid=True,
                ),
            ],
        )
        with pytest.raises(ValueError):
            check_nchoosek_constraints_as_bounds(domain)


def test_nchoosek_constraints_as_bounds():
    # define domain: no NChooseK constraints
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=-1, upper_bound=1)
            for i in range(5)
        ],
        output_features=[ContinuousOutput(key="y")],
    )
    bounds = nchoosek_constraints_as_bounds(domain, n_experiments=4)
    assert len(bounds) == 20
    _bounds = [
        (p.lower_bound, p.upper_bound)
        for p in domain.input_features
        if isinstance(p, ContinuousInput)
    ] * 4
    for i in range(20):
        assert _bounds[i] == bounds[i]

    # define domain: with NChooseK constraints
    # define domain: no NChooseK constraints
    # domain = Domain(
    #     input_features=[
    #         ContinuousInput(key=f"x{i+1}", lower_bound=-1, upper_bound=1)
    #         for i in range(5)
    #     ],
    #     output_features=[ContinuousOutput(key="y")],
    #     constraints=[opti.NChooseK(["x1", "x2", "x3"], max_active=1)],
    # )
    # np.random.seed(1)
    # bounds = nchoosek_constraints_as_bounds(domain, n_experiments=4)
    # _bounds = [
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    #     (0.0, 0.0),
    #     (-1.0, 1.0),
    #     (0.0, 0.0),
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    #     (0.0, 0.0),
    #     (0.0, 0.0),
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    #     (-1.0, 1.0),
    # ]
    # assert len(bounds) == 20
    # for i in range(20):
    #     assert _bounds[i] == bounds[i]
