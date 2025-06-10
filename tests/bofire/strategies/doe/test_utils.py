import importlib.util
import sys

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, NonlinearConstraint

from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.doe.objective import DOptimalityCriterion, get_objective_function
from bofire.strategies.doe.utils import (
    ConstraintWrapper,
    _minimize,
    check_nchoosek_constraints_as_bounds,
    constraints_as_scipy_constraints,
    convert_formula_to_string,
    get_formula_from_string,
    n_zero_eigvals,
    nchoosek_constraints_as_bounds,
)


CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None


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
    assert n_zero_eigvals(domain, "linear") == 1
    assert n_zero_eigvals(domain, "linear-and-quadratic") == 2
    assert n_zero_eigvals(domain, "linear-and-interactions") == 3
    assert n_zero_eigvals(domain, "fully-quadratic") == 7

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


def test_check_nchoosek_constraints_as_bounds():
    # define domain: possible to formulate as bounds, no NChooseK constraints
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(4)],
        outputs=[ContinuousOutput(key="y")],
    )
    check_nchoosek_constraints_as_bounds(domain)

    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(4)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[],
    )
    check_nchoosek_constraints_as_bounds(domain)

    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(4)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(features=["x1", "x2"], coefficients=[1, 1], rhs=0),
        ],
    )
    check_nchoosek_constraints_as_bounds(domain)

    # n-choose-k constraints when variables can be negative
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{1}", bounds=(0, 1)),
            ContinuousInput(key=f"x{2}", bounds=(0, 2)),
            ContinuousInput(key=f"x{3}", bounds=(0, 3)),
            ContinuousInput(key=f"x{4}", bounds=(0, 4)),
        ],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(features=["x1", "x2"], coefficients=[1, 1], rhs=0),
            LinearInequalityConstraint(
                features=["x3", "x4"],
                coefficients=[1, 1],
                rhs=0,
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

    # It should be allowed to have n-choose-k constraints when 0 is not in the bounds.
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0.1, 1)) for i in range(4)],
        outputs=[ContinuousOutput(key="y")],
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
        check_nchoosek_constraints_as_bounds(domain)  # FIXME: should be allowed

    # It should be allowed to have n-choose-k constraints when 0 is not in the bounds.
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key=f"x{1}", bounds=(0.1, 1.0)),
            ContinuousInput(key=f"x{2}", bounds=(0.1, 1.0)),
            ContinuousInput(key=f"x{3}", bounds=(0.1, 1.0)),
            ContinuousInput(key=f"x{4}", bounds=(0.1, 1.0)),
        ],
        outputs=[ContinuousOutput(key="y")],
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
        check_nchoosek_constraints_as_bounds(domain)  # FIXME: should be allowed

    # Not allowed: names parameters of two NChooseK overlap
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(4)],
        outputs=[ContinuousOutput(key="y")],
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


if __name__ == "__main__":
    get_formula_from_string_recursion_limit()
