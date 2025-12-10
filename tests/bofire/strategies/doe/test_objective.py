import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch
from formulaic import Formula

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.linear import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.strategies.doe import (
    AOptimalityCriterion,
    DOptimalityCriterion,
    EOptimalityCriterion,
    GOptimalityCriterion,
    IOptimalityCriterion,
    SpaceFillingCriterion,
)
from bofire.strategies.api import DoEStrategy
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


def test_tensor_to_model_matrix():
    """Test the tensor_to_model_matrix method of ModelBasedObjective."""
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i + 1}", bounds=(0, 1)) for i in range(3)],
        outputs=[ContinuousOutput(key="y")],
    )

    # Test with linear formula
    formula = get_formula_from_string("linear", inputs=domain.inputs)
    d_optimality = DOptimality(
        domain=domain,
        formula=formula,
        n_experiments=2,
    )

    # Create a test design matrix tensor
    D = torch.tensor([[0.5, 0.3, 0.7], [0.2, 0.8, 0.4]], dtype=torch.float64)

    # Get model matrix using our method
    X = d_optimality.tensor_to_model_matrix(D)

    # Verify the shape - should be [n_experiments, n_model_terms]
    # Linear formula with 3 inputs: 1 + x1 + x2 + x3 = 4 terms
    assert X.shape == (2, 4)

    # Verify the first column is all ones (intercept)
    assert torch.allclose(X[:, 0], torch.ones(2, dtype=torch.float64))

    # Verify the other columns match the design matrix
    assert torch.allclose(X[:, 1], D[:, 0])  # x1
    assert torch.allclose(X[:, 2], D[:, 1])  # x2
    assert torch.allclose(X[:, 3], D[:, 2])  # x3

    # Test with quadratic formula
    formula = Formula("x1 + x2 + {x1**2} + x1:x2")
    d_optimality = DOptimality(
        domain=domain,
        formula=formula,
        n_experiments=2,
    )

    X = d_optimality.tensor_to_model_matrix(D)

    # Should have 5 terms: 1 + x1 + x2 + x1^2 + x1:x2
    assert X.shape == (2, 5)
    assert torch.allclose(X[:, 0], torch.ones(2, dtype=torch.float64))  # intercept
    assert torch.allclose(X[:, 1], D[:, 0])  # x1
    assert torch.allclose(X[:, 2], D[:, 1])  # x2
    assert torch.allclose(X[:, 3], D[:, 0] ** 2)  # x1^2
    assert torch.allclose(X[:, 4], D[:, 0] * D[:, 1])  # x1:x2


def test_get_candidate_fim_rank():
    """Test the get_candidate_fim_rank method of DoEStrategy."""
    # Create a simple domain with 3 continuous inputs
    simple_domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    # Test 1: No candidates should return 0
    data_model = data_models.DoEStrategy(
        domain=simple_domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    assert strategy.get_candidate_fim_rank() == 0

    # Test 2: Full rank Fisher Information Matrix (4 candidates for linear model: intercept + 3 variables)
    candidates_full_rank = pd.DataFrame(
        {
            "x1": [1.0, 0.0, 0.0, 0.5],
            "x2": [0.0, 1.0, 0.0, 0.5],
            "x3": [0.0, 0.0, 1.0, 0.5],
        }
    )
    strategy.set_candidates(candidates_full_rank)
    rank = strategy.get_candidate_fim_rank()
    assert rank == 4  # Intercept + 3 variables = 4 estimable parameters

    # Test 3: Rank-deficient Fisher Information Matrix (linearly dependent design points)
    candidates_rank_deficient = pd.DataFrame(
        {
            "x1": [
                1.0,
                1.0,
                0.0,
                0.0,
            ],  # Two pairs of identical rows creates linear dependence in design matrix
            "x2": [0.0, 0.0, 1.0, 1.0],
            "x3": [
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # Constant, provides no information beyond intercept
        }
    )
    strategy.set_candidates(candidates_rank_deficient)
    rank = strategy.get_candidate_fim_rank()
    assert (
        rank == 2
    )  # Only 2 linearly independent design points: spans intercept + 1 direction

    # Test 4: Test with quadratic formula
    data_model_quad = data_models.DoEStrategy(
        domain=simple_domain, criterion=DOptimalityCriterion(formula="fully-quadratic")
    )
    strategy_quad = DoEStrategy(data_model_quad)
    strategy_quad.set_candidates(candidates_full_rank)
    rank_quad = strategy_quad.get_candidate_fim_rank()
    # Fully quadratic has 10 terms (excluding intercept), with 4 candidates rank is 4
    assert rank_quad == 4

    # Test 5: SpaceFilling criterion should raise error
    data_model_space = data_models.DoEStrategy(
        domain=simple_domain, criterion=SpaceFillingCriterion()
    )
    strategy_space = DoEStrategy(data_model_space)
    strategy_space.set_candidates(candidates_full_rank)

    with pytest.raises(
        ValueError,
        match="get_candidate_fim_rank\\(\\) only works with DoEOptimalityCriterion",
    ):
        strategy_space.get_candidate_fim_rank()


def test_get_candidate_fim_rank_categorical_discrete():
    """Test the get_candidate_fim_rank method with categorical and discrete inputs."""
    # Create a domain with mixed input types
    mixed_domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            DiscreteInput(key="x2", values=[0.1, 0.5, 1.0]),
            CategoricalInput(key="x3", categories=["A", "B", "C"]),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    # Test 1: No candidates should return 0
    data_model = data_models.DoEStrategy(
        domain=mixed_domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)
    assert strategy.get_candidate_fim_rank() == 0

    # Test 2: Mixed input candidates
    candidates_mixed = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.5, 0.2],
            "x2": [0.1, 0.5, 1.0, 0.5],  # Discrete values
            "x3": ["A", "B", "C", "A"],  # Categorical values
        }
    )
    strategy.set_candidates(candidates_mixed)
    rank = strategy.get_candidate_fim_rank()
    # Actual rank depends on linear independence in the transformed design matrix
    assert rank == 3

    # Test 3: Test with interactions formula for mixed types
    data_model_interactions = data_models.DoEStrategy(
        domain=mixed_domain,
        criterion=DOptimalityCriterion(formula="linear-and-interactions"),
    )
    strategy_interactions = DoEStrategy(data_model_interactions)
    strategy_interactions.set_candidates(candidates_mixed)
    rank_interactions = strategy_interactions.get_candidate_fim_rank()
    # With interactions, rank is 4 (limited by number of candidates)
    assert rank_interactions == 4

    # Test 4: Rank-deficient case with repeated categorical/discrete values
    candidates_repeated = pd.DataFrame(
        {
            "x1": [0.0, 0.0, 0.5, 0.5],  # Repeated continuous values
            "x2": [0.1, 0.1, 0.5, 0.5],  # Repeated discrete values
            "x3": ["A", "A", "B", "B"],  # Repeated categorical values
        }
    )
    strategy.set_candidates(candidates_repeated)
    rank_repeated = strategy.get_candidate_fim_rank()
    assert (
        rank_repeated == 2
    )  # Only 2 unique design points: intercept + 1 independent direction

    # Test 5: Single categorical level (should reduce rank)
    candidates_single_cat = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.5, 0.2],
            "x2": [0.1, 0.5, 1.0, 0.1],
            "x3": ["A", "A", "A", "A"],  # All same category
        }
    )
    strategy.set_candidates(candidates_single_cat)
    rank_single = strategy.get_candidate_fim_rank()
    # Intercept + x1 + x2 (x3 categorical doesn't vary, contributes no information)
    assert rank_single == 3


def test_get_candidate_fim_rank_vs_required_experiments():
    """Test that Fisher Information Matrix rank is at most the required number of experiments.
    Also tests the get_additional_experiments_needed method."""
    # Test with continuous inputs only
    continuous_domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            ContinuousInput(key="x3", bounds=(0, 1)),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    for formula in ["linear", "linear-and-interactions", "fully-quadratic"]:
        data_model = data_models.DoEStrategy(
            domain=continuous_domain, criterion=DOptimalityCriterion(formula=formula)
        )
        strategy = DoEStrategy(data_model=data_model)

        required_experiments = strategy.get_required_number_of_experiments()

        # Create candidates with more experiments than required
        n_candidates = required_experiments + 5 if required_experiments else 10
        candidates = pd.DataFrame(
            {
                "x1": np.random.uniform(0, 1, n_candidates),
                "x2": np.random.uniform(0, 1, n_candidates),
                "x3": np.random.uniform(0, 1, n_candidates),
            }
        )

        strategy.set_candidates(candidates)
        fim_rank = strategy.get_candidate_fim_rank()

        # Fisher Information Matrix rank should be at most the required number of experiments
        if required_experiments is not None:
            assert (
                fim_rank <= required_experiments
            ), f"FIM rank ({fim_rank}) exceeds required experiments ({required_experiments}) for {formula}"

        # Also should be at most the number of candidates
        assert (
            fim_rank <= n_candidates
        ), f"FIM rank ({fim_rank}) exceeds number of candidates ({n_candidates}) for {formula}"

        # Test the new get_additional_experiments_needed method
        if required_experiments is not None:
            # With default epsilon=3
            additional_needed = strategy.get_additional_experiments_needed()
            difference = required_experiments - fim_rank
            expected_additional = 3 if difference == 0 else difference
            assert (
                additional_needed == expected_additional
            ), f"Additional experiments mismatch for {formula}: got {additional_needed}, expected {expected_additional}"

            # With custom epsilon=5
            additional_needed_custom = strategy.get_additional_experiments_needed(
                epsilon=5
            )
            expected_additional_custom = 5 if difference == 0 else difference
            assert (
                additional_needed_custom == expected_additional_custom
            ), f"Additional experiments (epsilon=5) mismatch for {formula}: got {additional_needed_custom}, expected {expected_additional_custom}"

    # Test with mixed input types
    mixed_domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            DiscreteInput(key="x2", values=[0.1, 0.5, 1.0]),
            CategoricalInput(key="x3", categories=["A", "B"]),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    data_model = data_models.DoEStrategy(
        domain=mixed_domain, criterion=DOptimalityCriterion(formula="linear")
    )
    strategy = DoEStrategy(data_model=data_model)

    required_experiments = strategy.get_required_number_of_experiments()

    candidates_mixed = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.5, 0.2, 0.8, 0.3],
            "x2": [0.1, 0.5, 1.0, 0.5, 0.1, 1.0],
            "x3": ["A", "B", "A", "B", "A", "B"],
        }
    )

    strategy.set_candidates(candidates_mixed)
    fim_rank = strategy.get_candidate_fim_rank()

    if required_experiments is not None:
        assert (
            fim_rank <= required_experiments
        ), f"Mixed domain: FIM rank ({fim_rank}) exceeds required experiments ({required_experiments})"

        # Test get_additional_experiments_needed with mixed inputs
        additional_needed = strategy.get_additional_experiments_needed()
        difference = required_experiments - fim_rank
        expected_additional = 3 if difference == 0 else difference
        assert (
            additional_needed == expected_additional
        ), f"Mixed domain: Additional experiments mismatch: got {additional_needed}, expected {expected_additional}"

        # Test with epsilon=0 (no buffer)
        additional_no_buffer = strategy.get_additional_experiments_needed(epsilon=0)
        expected_no_buffer = 0 if difference == 0 else difference
        assert (
            additional_no_buffer == expected_no_buffer
        ), f"Mixed domain: Additional experiments (no buffer) mismatch: got {additional_no_buffer}, expected {expected_no_buffer}"
