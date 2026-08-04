import importlib
import importlib.util

import numpy as np
import pytest
import torch
from formulaic import Formula

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
from bofire.strategies.doe.utils_categorical_discrete import create_continuous_domain


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


def test_tensor_to_model_matrix_categorical_discrete():
    """Test tensor_to_model_matrix with categorical and discrete inputs.

    This test verifies that when using categorical/discrete inputs, the model matrix
    is built correctly using the relaxed domain with auxiliary variables, but the
    formula is based on the original domain inputs.
    """
    # Create original domain with mixed input types
    original_domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            DiscreteInput(key="x2", values=[0.1, 0.5, 1.0]),
            CategoricalInput(key="x3", categories=["A", "B", "C"]),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    # Create relaxed domain with auxiliary variables
    relaxed_domain, *_ = create_continuous_domain(domain=original_domain)

    # Verify that relaxed domain has more inputs than original
    # Original: x1, x2, x3 (3 inputs)
    # Relaxed: x1, x2, x3_A, x3_B, x3_C, aux vars for discrete (more inputs)
    assert len(relaxed_domain.inputs) > len(original_domain.inputs)

    # Test 1: Linear formula based on ORIGINAL domain inputs
    formula_original = get_formula_from_string("linear", inputs=original_domain.inputs)

    # Create objective using RELAXED domain but formula from ORIGINAL inputs
    d_optimality = DOptimality(
        domain=relaxed_domain,
        formula=formula_original,
        n_experiments=2,
    )

    # Create a relaxed design tensor (includes all auxiliary variables)
    # Order in relaxed domain: x2, x3_A, x3_B, x3_C, x1, aux_vars
    n_relaxed_inputs = len(relaxed_domain.inputs)
    D_relaxed = torch.zeros((2, n_relaxed_inputs), dtype=torch.float64)

    # Set values for the relaxed variables
    # For simplicity, let's set:
    # Row 1: x1=0.3, x2=0.5, x3="A" (x3_A=1, x3_B=0, x3_C=0)
    # Row 2: x1=0.7, x2=1.0, x3="B" (x3_A=0, x3_B=1, x3_C=0)

    # Find indices in relaxed domain
    relaxed_keys = relaxed_domain.inputs.get_keys()
    idx_x1 = relaxed_keys.index("x1")
    idx_x2 = relaxed_keys.index("x2")
    idx_aux_x3_A = relaxed_keys.index("aux_x3_A")
    idx_aux_x3_B = relaxed_keys.index("aux_x3_B")
    idx_aux_x3_C = relaxed_keys.index("aux_x3_C")

    # Row 1: x1=0.3, x2=0.5, x3="A" (aux_x3_A=1, aux_x3_B=0, aux_x3_C=0)
    D_relaxed[0, idx_x1] = 0.3
    D_relaxed[0, idx_x2] = 0.5
    D_relaxed[0, idx_aux_x3_A] = 1.0
    D_relaxed[0, idx_aux_x3_B] = 0.0
    D_relaxed[0, idx_aux_x3_C] = 0.0

    # Row 2: x1=0.7, x2=1.0, x3="B" (aux_x3_A=0, aux_x3_B=1, aux_x3_C=0)
    D_relaxed[1, idx_x1] = 0.7
    D_relaxed[1, idx_x2] = 1.0
    D_relaxed[1, idx_aux_x3_A] = 0.0
    D_relaxed[1, idx_aux_x3_B] = 1.0
    D_relaxed[1, idx_aux_x3_C] = 0.0

    # Get model matrix
    X = d_optimality.tensor_to_model_matrix(D_relaxed)

    # For linear formula with 3 original inputs (x1, x2, x3):
    # Expected terms: 1 + x1 + x2 + x3[T.B] + x3[T.C]
    # (categorical x3 with 3 levels uses 2 dummy variables)
    # So we expect 5 columns
    assert X.shape[0] == 2  # 2 experiments
    assert X.shape[1] == 5  # intercept + x1 + x2 + 2 categorical dummies

    # Check intercept
    assert torch.allclose(X[:, 0], torch.ones(2, dtype=torch.float64))

    # Check x1 values
    assert torch.allclose(X[:, 1], torch.tensor([0.3, 0.7], dtype=torch.float64))

    # Check x2 values
    assert torch.allclose(X[:, 2], torch.tensor([0.5, 1.0], dtype=torch.float64))

    # Check categorical dummies (aux_x3_A and aux_x3_B, with C as reference)
    # Row 1: x3="A" -> aux_x3_A=1, aux_x3_B=0
    # Row 2: x3="B" -> aux_x3_A=0, aux_x3_B=1
    assert torch.allclose(
        X[:, 3], torch.tensor([1.0, 0.0], dtype=torch.float64)
    )  # aux_x3_A
    assert torch.allclose(
        X[:, 4], torch.tensor([0.0, 1.0], dtype=torch.float64)
    )  # aux_x3_B

    # Test 2: Linear + interactions formula
    formula_interactions = get_formula_from_string(
        "linear-and-interactions", inputs=original_domain.inputs
    )

    d_optimality_interactions = DOptimality(
        domain=relaxed_domain,
        formula=formula_interactions,
        n_experiments=2,
    )

    X_interactions = d_optimality_interactions.tensor_to_model_matrix(D_relaxed)

    # With interactions, we should have more columns than the linear model
    # Base terms: 1 + x1 + x2 + aux_x3_A + aux_x3_B = 5
    # Interactions: x1:aux_x3_A, x2:aux_x3_A, x1:aux_x3_B, x2:aux_x3_B, x1:x2 = 5 more
    # Total = 10 columns
    assert X_interactions.shape[0] == 2
    assert X_interactions.shape[1] == 10

    # Verify that the first 5 columns (base terms) match the linear model
    assert torch.allclose(X_interactions[:, :5], X)

    # Verify the exact ordering of interaction terms (columns 5-9)
    # Row 1: x1=0.3, x2=0.5, aux_x3_A=1.0, aux_x3_B=0.0
    # Row 2: x1=0.7, x2=1.0, aux_x3_A=0.0, aux_x3_B=1.0

    # Actual order based on formula generation:
    # Column 5: x1:aux_x3_A
    assert torch.allclose(
        X_interactions[:, 5], torch.tensor([0.3 * 1.0, 0.7 * 0.0], dtype=torch.float64)
    )

    # Column 6: x2:aux_x3_A
    assert torch.allclose(
        X_interactions[:, 6], torch.tensor([0.5 * 1.0, 1.0 * 0.0], dtype=torch.float64)
    )

    # Column 7: x1:aux_x3_B
    assert torch.allclose(
        X_interactions[:, 7], torch.tensor([0.3 * 0.0, 0.7 * 1.0], dtype=torch.float64)
    )

    # Column 8: x2:aux_x3_B
    assert torch.allclose(
        X_interactions[:, 8], torch.tensor([0.5 * 0.0, 1.0 * 1.0], dtype=torch.float64)
    )

    # Column 9: x1:x2
    assert torch.allclose(
        X_interactions[:, 9], torch.tensor([0.3 * 0.5, 0.7 * 1.0], dtype=torch.float64)
    )
