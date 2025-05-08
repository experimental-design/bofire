import importlib

import numpy as np
import pytest
from formulaic import Formula

from bofire.data_models.constraints.linear import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.doe import (
    AOptimalityCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    GOptimalityCriterion,
    IOptimalityCriterion,
    SpaceFillingCriterion,
)
from bofire.strategies.doe.objective import (
    AOptimality,
    DOptimality,
    EOptimality,
    GOptimality,
    IOptimality,
    SpaceFilling,
    get_objective_function,
)
from bofire.strategies.doe.utils import get_formula_from_string


CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None


def test_DOptimality_instantiation():
    # default jacobian building block
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i + 1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    formula = Formula("x1 + x2 + x3 + x1:x2 + {x3**2}")

    d_optimality = DOptimality(
        domain=domain,
        formula=formula,
        n_experiments=2,
    )

    assert isinstance(d_optimality.domain, Domain)
    assert all(
        np.array(d_optimality.domain.inputs.get_keys()) == np.array(["x1", "x2", "x3"]),
    )
    for i in d_optimality.domain.inputs.get():
        assert isinstance(i, ContinuousInput)
        assert i.upper_bound == 1
        assert i.lower_bound == 0
    assert all(np.array(d_optimality.domain.outputs.get_keys()) == np.array(["y"]))

    assert isinstance(d_optimality.formula, Formula)
    assert all(
        np.array(d_optimality.formula, dtype=str)
        == np.array(["1", "x1", "x2", "x3", "x3 ** 2", "x1:x2"]),
    )

    assert np.shape(
        d_optimality.evaluate_jacobian(np.array([[1, 1, 1], [2, 2, 2]]).flatten()),
    ) == (6,)

    assert np.shape(
        d_optimality.evaluate_hessian(np.array([[1, 1, 1], [2, 2, 2]]).flatten()),
    ) == (6, 6)

    # 5th order model
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i + 1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    formula = Formula("{x1**5} + {x2**5} + {x3**5}")

    d_optimality = DOptimality(
        domain=domain,
        formula=formula,
        n_experiments=3,
    )

    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    B = np.zeros(shape=(3, 4))
    B[:, 1:] = 5 * np.diag(x[0] ** 4)

    assert np.shape(
        d_optimality.evaluate_jacobian(
            np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).flatten(),
        ),
    ) == (9,)

    assert np.shape(
        d_optimality.evaluate_hessian(
            np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).flatten()
        ),
    ) == (9, 9)


def test_DOptimality_evaluate_jacobian():
    # n_experiment = 1, n_inputs = 2, model: x1 + x2
    def get_jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:  # type: ignore
        return -2 * x / (x[0] ** 2 + x[1] ** 2 + delta)

    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i + 1}",
                bounds=(0, 1),
            )
            for i in range(2)
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    formula = Formula("x1 + x2 - 1")
    n_experiments = 1
    d_optimality = DOptimality(
        domain=domain,
        formula=formula,
        n_experiments=n_experiments,
        delta=1e-3,
    )

    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(2)
        assert np.allclose(
            d_optimality.evaluate_jacobian(x), get_jacobian(x), rtol=1e-3
        )

    # n_experiment = 1, n_inputs = 2, formula: x1**2 + x2**2
    def get_jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:  # type: ignore
        return -4 * x**3 / (x[0] ** 4 + x[1] ** 4 + delta)

    formula = Formula("{x1**2} + {x2**2} - 1")
    d_optimality = DOptimality(
        domain=domain,
        formula=formula,
        n_experiments=n_experiments,
        delta=1e-3,
    )
    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(2)
        assert np.allclose(
            d_optimality.evaluate_jacobian(x), get_jacobian(x), rtol=1e-3
        )

    # n_experiment = 2, n_inputs = 2, formula = x1 + x2
    def get_jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:
        X = x.reshape(2, 2)

        y = np.empty(4)
        y[0] = (
            -2
            * (
                x[0] * (x[1] ** 2 + x[3] ** 2 + delta)
                - x[1] * (x[0] * x[1] + x[2] * x[3])
            )
            / np.linalg.det(X.T @ X + delta * np.eye(2))
        )
        y[1] = (
            -2
            * (
                x[1] * (x[0] ** 2 + x[2] ** 2 + delta)
                - x[0] * (x[0] * x[1] + x[2] * x[3])
            )
            / np.linalg.det(X.T @ X + delta * np.eye(2))
        )
        y[2] = (
            -2
            * (
                x[2] * (x[1] ** 2 + x[3] ** 2 + delta)
                - x[3] * (x[0] * x[1] + x[2] * x[3])
            )
            / np.linalg.det(X.T @ X + delta * np.eye(2))
        )
        y[3] = (
            -2
            * (
                x[3] * (x[0] ** 2 + x[2] ** 2 + delta)
                - x[2] * (x[0] * x[1] + x[2] * x[3])
            )
            / np.linalg.det(X.T @ X + delta * np.eye(2))
        )

        return y

    formula = Formula("x1 + x2 - 1")
    n_experiments = 2
    d_optimality = DOptimality(
        domain=domain,
        formula=formula,
        n_experiments=n_experiments,
        delta=1e-3,
    )
    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(4)
        assert np.allclose(
            d_optimality.evaluate_jacobian(x), get_jacobian(x), rtol=1e-3
        )

    # n_experiment = 2, n_inputs = 2, formula = x1**2 + x2**2
    def jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:
        X = x.reshape(2, 2) ** 2

        y = np.empty(4)
        y[0] = (
            -4
            * (
                x[0] ** 3 * (x[1] ** 4 + x[3] ** 4 + delta)
                - x[0] * x[1] ** 2 * (x[0] ** 2 * x[1] ** 2 + x[2] ** 2 * x[3] ** 2)
            )
            / np.linalg.det(X.T @ X + delta * np.eye(2))
        )
        y[1] = (
            -4
            * (
                x[1] ** 3 * (x[0] ** 4 + x[2] ** 4 + delta)
                - x[1] * x[0] ** 2 * (x[0] ** 2 * x[1] ** 2 + x[2] ** 2 * x[3] ** 2)
            )
            / np.linalg.det(X.T @ X + delta * np.eye(2))
        )
        y[2] = (
            -4
            * (
                x[2] ** 3 * (x[1] ** 4 + x[3] ** 4 + delta)
                - x[2] * x[3] ** 2 * (x[0] ** 2 * x[1] ** 2 + x[2] ** 2 * x[3] ** 2)
            )
            / np.linalg.det(X.T @ X + delta * np.eye(2))
        )
        y[3] = (
            -4
            * (
                x[3] ** 3 * (x[0] ** 4 + x[2] ** 4 + delta)
                - x[3] * x[2] ** 2 * (x[0] ** 2 * x[1] ** 2 + x[2] ** 2 * x[3] ** 2)
            )
            / np.linalg.det(X.T @ X + delta * np.eye(2))
        )

        return y

    formula = Formula("{x1**2} + {x2**2} - 1")
    d_optimality = DOptimality(
        domain=domain,
        formula=formula,
        n_experiments=n_experiments,
        delta=1e-3,
    )

    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(4)
        assert np.allclose(d_optimality.evaluate_jacobian(x), jacobian(x), rtol=1e-3)


def test_DOptimality_evaluate():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i + 1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    formula = get_formula_from_string("linear", inputs=domain.inputs)

    d_optimality = DOptimality(
        domain=domain, formula=formula, n_experiments=3, delta=1e-7
    )
    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)
    assert np.allclose(d_optimality.evaluate(x), -np.log(4) - np.log(1e-7))


def test_AOptimality_evaluate():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i + 1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    formula = get_formula_from_string("linear", inputs=domain.inputs)

    a_optimality = AOptimality(domain=domain, formula=formula, n_experiments=4)

    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    assert np.allclose(a_optimality.evaluate(x), 3 * 1.9999991 + 0.9999996)


def test_AOptimality_evaluate_jacobian():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    formula = get_formula_from_string("linear", inputs=domain.inputs)

    a_optimality = AOptimality(domain=domain, formula=formula, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    def grad(x):
        return (
            2
            * np.array([-(x[0] * x[1] + x[1] ** 2 + 2), (x[0] * x[1] + x[0] ** 2 + 2)])
            / (x[0] - x[1]) ** 3
        )

    assert np.allclose(a_optimality.evaluate_jacobian(x), grad(x))


def test_EOptimality_evaluate():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    formula = get_formula_from_string("linear", inputs=domain.inputs)

    e_optimality = EOptimality(domain=domain, formula=formula, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    min_eigval = 0.5 * (
        x[0] ** 2
        - np.sqrt(
            x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + 8 * x[0] * x[1] + x[1] ** 4 + 4,
        )
        + x[1] ** 2
        + 2
    )

    assert np.allclose(e_optimality.evaluate(x), -min_eigval)


def test_EOptimality_evaluate_jacobian():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    formula = get_formula_from_string("linear", inputs=domain.inputs)

    e_optimality = EOptimality(domain=domain, formula=formula, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    def grad(x):
        temp = np.sqrt(
            x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + 8 * x[0] * x[1] + x[1] ** 4 + 4,
        )
        return np.array(
            [
                (x[0] ** 3 + x[0] * x[1] ** 2 + 2 * x[1]) / temp - x[0],
                (x[1] ** 3 + x[1] * x[0] ** 2 + 2 * x[0]) / temp - x[1],
            ],
        )

    assert np.allclose(e_optimality.evaluate_jacobian(x), grad(x))


def test_GOptimality_evaluate():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    formula = get_formula_from_string("linear", inputs=domain.inputs)

    g_optimality = GOptimality(domain=domain, formula=formula, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    # all eigenvalues are 1 since A = [[1,1],[1,0.5]] is invertible and therefore H = A (A.T A)^-1 A.T = A A^-1 A.T^-1 A.T = 1

    assert np.allclose(g_optimality.evaluate(x), 1)


def test_GOptimality_evaluate_jacobian():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    formula = get_formula_from_string("linear", inputs=domain.inputs)

    g_optimality = GOptimality(domain=domain, formula=formula, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    # all eigenvalues are 1 since A = [[1,1],[1,0.5]] is invertible and therefore H = A (A.T A)^-1 A.T = A A^-1 A.T^-1 A.T = 1
    # thus, the jacobian vanishes.

    assert np.allclose(g_optimality.evaluate_jacobian(x), np.zeros(2))


def test_SpaceFilling_evaluate():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )

    space_filling = SpaceFilling(domain=domain, n_experiments=4, delta=0)

    x = np.array([1, 0.6, 0.1, 0.3])

    assert np.allclose(space_filling.evaluate(x), -1.4)


def test_SpaceFilling_evaluate_jacobian():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )

    space_filling = SpaceFilling(domain=domain, n_experiments=4, delta=0)

    x = np.array([1, 0.4, 0, 0.1])

    assert np.allclose(space_filling.evaluate_jacobian(x), [-1, -1, 2, 0])


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_MinMaxTransform():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    x = np.array([1, 0.8, 0.55, 0.65])
    x_scaled = x * 2 - 1

    for cls in [
        DOptimalityCriterion,
        AOptimalityCriterion,
        EOptimalityCriterion,
        GOptimalityCriterion,
        SpaceFillingCriterion,
    ]:
        if cls == SpaceFillingCriterion:
            objective_unscaled = get_objective_function(
                cls(
                    transform_range=None,
                ),
                domain=domain,
                n_experiments=4,
            )

            objective_scaled = get_objective_function(
                cls(
                    transform_range=(-1.0, 1.0),
                ),
                domain=domain,
                n_experiments=4,
            )
        else:
            objective_unscaled = get_objective_function(
                cls(
                    formula="linear",
                    delta=0,
                    transform_range=None,
                ),
                domain=domain,
                n_experiments=4,
            )

            objective_scaled = get_objective_function(
                cls(
                    formula="linear",
                    delta=0,
                    transform_range=(-1.0, 1.0),
                ),
                domain=domain,
                n_experiments=4,
            )
        assert np.allclose(
            objective_unscaled.evaluate(x_scaled),
            objective_scaled.evaluate(x),
        )
        assert np.allclose(
            2 * objective_unscaled.evaluate_jacobian(x_scaled),
            objective_scaled.evaluate_jacobian(x),
        )

        objective_unscaled = get_objective_function(
            IOptimalityCriterion(
                formula="linear",
                delta=0,
                transform_range=None,
                n_space_filling_points=4,
                ipopt_options={"max_iter": 200},
            ),
            domain=domain,
            n_experiments=4,
        )
        with pytest.raises(ValueError):
            objective_scaled = get_objective_function(
                IOptimalityCriterion(
                    formula="linear",
                    delta=0,
                    transform_range=(-1.0, 1.0),
                    n_space_filling_points=4,
                    ipopt_options={"max_iter": 200},
                ),
                domain=domain,
                n_experiments=4,
            )


@pytest.mark.skipif(not CYIPOPT_AVAILABLE, reason="requires cyipopt")
def test_IOptimality_instantiation():
    # no constraints
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )

    formula = get_formula_from_string("linear", inputs=domain.inputs)

    i_optimality = IOptimality(
        domain=domain,
        formula=formula,
        n_experiments=2,
    )
    assert np.allclose(np.linspace(0, 1, 100), i_optimality.Y.to_numpy().flatten())

    # inequality constraints
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(2)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearInequalityConstraint(
                features=["x1", "x2"], coefficients=[1, 0], rhs=0.5
            )
        ],
    )

    formula = get_formula_from_string("linear", inputs=domain.inputs)

    i_optimality = IOptimality(
        domain=domain,
        formula=formula,
        n_experiments=2,
    )

    assert np.allclose(
        np.linspace(0, 1, 100)[:50],
        np.unique(i_optimality.Y.to_numpy()[:, 0]),
    )

    # equality constraints
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(2)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[
            LinearEqualityConstraint(features=["x1", "x2"], coefficients=[1, 1], rhs=1)
        ],
    )

    formula = get_formula_from_string("linear", inputs=domain.inputs)

    i_optimality = IOptimality(
        domain=domain,
        formula=formula,
        n_experiments=2,
    )

    assert np.allclose(domain.constraints(i_optimality.Y), 0.0)
