import importlib.util
import sys

import numpy as np
import pytest
from scipy import sparse

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
from bofire.strategies.doe.objective import DOptimalityCriterion, get_objective_function
from bofire.strategies.doe.utils import (
    ConstraintWrapper,
    _minimize,
    constraints_as_scipy_constraints,
    convert_formula_to_string,
    formula_str_to_fully_continuous,
    get_formula_from_string,
    n_zero_eigvals,
    nchoosek_constraints_as_bounds,
)


CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None
POUNCE_AVAILABLE = importlib.util.find_spec("pounce") is not None


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

    # Now returns old-style dict constraints with a sparse (coo) jac.
    # scipy convention: "eq" → g(x)==0, "ineq" → g(x)>=0.
    constraints = constraints_as_scipy_constraints(domain, n_experiments)
    n = len(domain.inputs) * n_experiments
    assert [c["type"] for c in constraints] == ["eq", "ineq", "ineq"]
    for c in constraints:
        J = c["jac"](np.zeros(n))
        assert sparse.issparse(J)
        assert J.shape == (n_experiments, n)

    # mixture equality (normalized): jac == block-diagonal A, fun(0) == -rhs/norm
    A_eq = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]) / np.sqrt(3)
    assert np.allclose(constraints[0]["jac"](np.zeros(n)).toarray(), A_eq)
    assert np.allclose(constraints[0]["fun"](np.zeros(n)), -np.ones(2) / np.sqrt(3))

    # first inequality (A x <= ub): jac == -A, fun(0) == ub == 3.9/||[5,4]||
    ub1 = 3.9 / np.linalg.norm([5, 4])
    r1 = np.array([5, 4, 0]) / np.linalg.norm([5, 4])
    A1 = np.zeros((2, 6))
    A1[0, :3], A1[1, 3:] = r1, r1
    assert np.allclose(constraints[1]["jac"](np.zeros(n)).toarray(), -A1)
    assert np.allclose(constraints[1]["fun"](np.zeros(n)), ub1 * np.ones(2))

    # domain with nonlinear constraints (g = x1**2 + x2**2 - 1)
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
    # equality: fun = g(x); inequality (g<=0): fun = -g(x). At x=1, g=[1,1].
    assert constraints[0]["type"] == "eq"
    assert np.allclose(constraints[0]["fun"](np.ones(6)), [1, 1])
    assert sparse.issparse(constraints[0]["jac"](np.ones(6)))
    assert constraints[1]["type"] == "ineq"
    assert np.allclose(constraints[1]["fun"](np.ones(6)), [-1, -1])

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
    # NChooseK is an upper-sided inequality (g<=0) → "ineq", fun = -g(x).
    assert len(constraints) == 1
    assert constraints[0]["type"] == "ineq"
    assert np.allclose(
        constraints[0]["fun"](np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])),
        [-2, 0, 0],
    )

    # domain with batch (interpoint) constraint
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
    assert constraints[0]["type"] == "eq"
    A = np.array(
        [
            [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0],
        ],
        dtype=float,
    )
    assert np.allclose(constraints[0]["jac"](np.zeros(15)).toarray(), A)
    assert np.allclose(constraints[0]["fun"](np.zeros(15)), np.zeros(3))


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


def _mixture_doe_setup(n_experiments=4):
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
                features=["x1", "x2", "x3"], coefficients=[1, 1, 1], rhs=1
            ),
        ],
    )
    objective_function = get_objective_function(
        criterion, domain=domain, n_experiments=n_experiments
    )
    x0 = (
        domain.inputs.sample(n=n_experiments, method=SamplingMethodEnum.UNIFORM, seed=1)
        .to_numpy()
        .flatten()
    )
    constraints = constraints_as_scipy_constraints(
        domain, n_experiments, ignore_nchoosek=True
    )
    bounds = nchoosek_constraints_as_bounds(domain, n_experiments)
    return objective_function, x0, bounds, constraints, n_experiments


@pytest.mark.parametrize("optimizer", ["ipopt", "pounce", "scipy"])
def test_minimize(optimizer):
    """Each backend reaches a feasible design of the same (D-optimal) quality."""
    if optimizer == "ipopt" and not CYIPOPT_AVAILABLE:
        pytest.skip("cyipopt required")
    if optimizer == "pounce" and not POUNCE_AVAILABLE:
        pytest.skip("pounce required")
    obj, x0, bounds, constraints, n = _mixture_doe_setup()

    ref = _minimize(
        obj,
        x0,
        bounds,
        constraints,
        optimizer="scipy",
        optimizer_options={"max_iter": 500},
    ).reshape(n, 3)
    res = _minimize(
        obj,
        x0,
        bounds,
        constraints,
        optimizer=optimizer,
        optimizer_options={"max_iter": 500, "print_level": 0},
    ).reshape(n, 3)

    # feasible (mixture sum = 1 per experiment) and same optimum value as scipy
    assert np.allclose(res.sum(axis=1), 1.0, atol=1e-6)
    assert abs(obj.evaluate(res.flatten()) - obj.evaluate(ref.flatten())) < 1e-3


def test_minimize_box_only_uses_lbfgsb():
    """A box-only DoE (no general constraints) solves regardless of `optimizer`
    (routes to scipy L-BFGS-B) and stays within bounds."""
    n_experiments = 4
    criterion = DOptimalityCriterion(formula="linear")
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in (1, 2, 3)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[],  # only box bounds
    )
    obj = get_objective_function(criterion, domain=domain, n_experiments=n_experiments)
    x0 = (
        domain.inputs.sample(n=n_experiments, method=SamplingMethodEnum.UNIFORM, seed=1)
        .to_numpy()
        .flatten()
    )
    constraints = constraints_as_scipy_constraints(
        domain, n_experiments, ignore_nchoosek=True
    )
    assert constraints == []  # box-only → no general constraints
    bounds = nchoosek_constraints_as_bounds(domain, n_experiments)

    # optimizer="pounce" but box-only → L-BFGS-B (no IPM/cyipopt/pounce needed)
    res = _minimize(
        obj, x0, bounds, constraints, optimizer="pounce", optimizer_options={}
    ).reshape(n_experiments, 3)
    assert (res >= -1e-6).all() and (res <= 1 + 1e-6).all()


def test_doestrategy_datamodel_legacy_kwargs_deprecated():
    """The public deprecation surface is the DoEStrategy data model: legacy
    `use_cyipopt`/`ipopt_options`/`use_hessian` are mapped to `optimizer`/
    `optimizer_options` with a DeprecationWarning."""
    from bofire.data_models.strategies.api import DoEStrategy as _DoEStrategyDM
    from bofire.data_models.strategies.api import DOptimalityCriterion as _DOptDM

    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in (1, 2, 3)],
        outputs=[ContinuousOutput(key="y")],
    )
    with pytest.warns(DeprecationWarning):
        dm = _DoEStrategyDM(
            domain=domain,
            criterion=_DOptDM(formula="linear"),
            use_cyipopt=False,
            ipopt_options={"max_iter": 200},
            use_hessian=True,
        )
    assert dm.optimizer == "scipy"
    assert dm.optimizer_options == {"max_iter": 200}
    assert not hasattr(dm, "use_cyipopt")


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
