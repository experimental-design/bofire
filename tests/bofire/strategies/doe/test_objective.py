import numpy as np
import pandas as pd
from formulaic import Formula

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.strategies.doe.objective import (
    AOptimality,
    DOptimality,
    EOptimality,
    GOptimality,
    Objective,
    SpaceFilling,
)
from bofire.strategies.doe.utils import get_formula_from_string


def test_Objective_model_jacobian_t():
    # "small" model
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    vars = domain.inputs.get_keys()
    f = Formula("x1 + x2 + x3 + x1:x2 + {x3**2}")
    x = np.array([[1, 2, 3]])

    objective = Objective(
        domain=domain,
        model=f,
        n_experiments=1,
    )
    model_jacobian_t = objective._model_jacobian_t

    B = np.zeros(shape=(3, 6))
    B[:, 1:4] = np.eye(3)
    B[:, 4] = np.array([0, 0, 6])
    B[:, 5] = np.array([2, 1, 0])

    assert np.allclose(B, model_jacobian_t(x))

    # fully quadratic model
    f = Formula("x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + {x1**2} + {x2**2} + {x3**2}")
    model_terms = np.array(f, dtype=str)
    x = np.array([[1, 2, 3]])

    objective = Objective(
        domain=domain,
        model=f,
        n_experiments=1,
    )
    model_jacobian_t = objective._model_jacobian_t
    B = np.zeros(shape=(3, 10))
    B[:, 1:4] = np.eye(3)
    B[:, 4:7] = 2 * np.diag(x[0])
    B[:, 7:] = np.array([[2, 1, 0], [3, 0, 1], [0, 3, 2]]).T
    B = pd.DataFrame(
        B,
        columns=[
            "1",
            "x1",
            "x2",
            "x3",
            "x1**2",
            "x2**2",
            "x3**2",
            "x1:x2",
            "x1:x3",
            "x2:x3",
        ],
    )
    B = B[model_terms].to_numpy()

    assert np.allclose(B, model_jacobian_t(x)[0])

    # fully cubic model
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(5)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    vars = ["x1", "x2", "x3", "x4", "x5"]
    n_vars = len(vars)

    formula = ""
    for name in vars:
        formula += name + " + "

    for name in vars:
        formula += "{" + name + "**2} + "
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            term = str(Formula(vars[j] + ":" + vars[i] + "-1")) + " + "
            formula += term

    for name in vars:
        formula += "{" + name + "**3} + "
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            for k in range(j + 1, n_vars):
                term = (
                    str(Formula(vars[k] + ":" + vars[j] + ":" + vars[i] + "-1")) + " + "
                )
                formula += term
    f = Formula(formula[:-3])
    x = np.array([[1, 2, 3, 4, 5]])
    objective = Objective(
        domain=domain,
        model=f,
        n_experiments=1,
    )
    model_jacobian_t = objective._model_jacobian_t

    B = np.array(
        [
            [
                0.0,
                1.0,
                2.0,
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                3.0,
                4.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                6.0,
                8.0,
                10.0,
                12.0,
                15.0,
                20.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                4.0,
                12.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                3.0,
                4.0,
                5.0,
                0.0,
                0.0,
                0.0,
                3.0,
                4.0,
                5.0,
                0.0,
                0.0,
                0.0,
                12.0,
                15.0,
                20.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                6.0,
                27.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                4.0,
                5.0,
                0.0,
                2.0,
                0.0,
                0.0,
                4.0,
                5.0,
                0.0,
                8.0,
                10.0,
                0.0,
                20.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                8.0,
                48.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                2.0,
                0.0,
                3.0,
                0.0,
                5.0,
                0.0,
                2.0,
                0.0,
                3.0,
                0.0,
                5.0,
                6.0,
                0.0,
                10.0,
                15.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                10.0,
                75.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                2.0,
                0.0,
                3.0,
                4.0,
                0.0,
                0.0,
                2.0,
                0.0,
                3.0,
                4.0,
                0.0,
                6.0,
                8.0,
                12.0,
            ],
        ]
    )

    B = pd.DataFrame(
        B,
        columns=[
            "1",
            "x1",
            "x1**2",
            "x1**3",
            "x2",
            "x2**2",
            "x2**3",
            "x3",
            "x3**2",
            "x3**3",
            "x4",
            "x4**2",
            "x4**3",
            "x5",
            "x5**2",
            "x5**3",
            "x2:x1",
            "x3:x1",
            "x4:x1",
            "x5:x1",
            "x3:x2",
            "x4:x2",
            "x5:x2",
            "x4:x3",
            "x5:x3",
            "x5:x4",
            "x3:x2:x1",
            "x4:x2:x1",
            "x5:x2:x1",
            "x4:x3:x1",
            "x5:x3:x1",
            "x5:x4:x1",
            "x4:x3:x2",
            "x5:x3:x2",
            "x5:x4:x2",
            "x5:x4:x3",
        ],
    )

    B = B[objective.model_terms].to_numpy()

    assert np.allclose(B, model_jacobian_t(x)[0])


def test_Objective_convert_input_to_model_tensor():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    model = get_formula_from_string("linear", domain=domain)

    d_optimality = DOptimality(domain=domain, model=model, n_experiments=3)
    x = np.array([1, 0, 0, 0, 2, 0, 0, 0, 3])
    print(domain.inputs)
    X = d_optimality._convert_input_to_model_tensor(x).detach().numpy()
    assert np.allclose(X, np.array([[1, 1, 0, 0], [1, 0, 2, 0], [1, 0, 0, 3]]))


def test_DOptimality_instantiation():
    # default jacobian building block
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    model = Formula("x1 + x2 + x3 + x1:x2 + {x3**2}")

    d_optimality = DOptimality(
        domain=domain,
        model=model,
        n_experiments=2,
    )

    assert isinstance(d_optimality.domain, Domain)
    assert all(
        np.array(d_optimality.domain.inputs.get_keys()) == np.array(["x1", "x2", "x3"])
    )
    for i in d_optimality.domain.inputs.get():
        assert isinstance(i, ContinuousInput)
        assert i.upper_bound == 1
        assert i.lower_bound == 0
    assert all(np.array(d_optimality.domain.outputs.get_keys()) == np.array(["y"]))

    assert isinstance(d_optimality.model, Formula)
    assert all(
        np.array(d_optimality.model, dtype=str)
        == np.array(["1", "x1", "x2", "x3", "x3**2", "x1:x2"])
    )

    x = np.array([[1, 2, 3], [1, 2, 3]])
    B = np.zeros(shape=(3, 6))
    B[:, 1:4] = np.eye(3)
    B[:, 4] = np.array([0, 0, 6])
    B[:, 5] = np.array([2, 1, 0])

    assert np.allclose(B, d_optimality._model_jacobian_t(x))
    assert np.shape(
        d_optimality.evaluate_jacobian(np.array([[1, 1, 1], [2, 2, 2]]).flatten())
    ) == (6,)

    # 5th order model
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    model = Formula("{x1**5} + {x2**5} + {x3**5}")

    d_optimality = DOptimality(
        domain=domain,
        model=model,
        n_experiments=3,
    )

    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    B = np.zeros(shape=(3, 4))
    B[:, 1:] = 5 * np.diag(x[0] ** 4)

    assert np.allclose(B, d_optimality._model_jacobian_t(x))
    assert np.shape(
        d_optimality.evaluate_jacobian(
            np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).flatten()
        )
    ) == (9,)


def test_DOptimality_evaluate_jacobian():
    # n_experiment = 1, n_inputs = 2, model: x1 + x2
    def jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:  # type: ignore
        return -2 * x / (x[0] ** 2 + x[1] ** 2 + delta)

    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(2)
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    model = Formula("x1 + x2 - 1")
    n_experiments = 1
    d_optimality = DOptimality(
        domain=domain, model=model, n_experiments=n_experiments, delta=1e-3
    )

    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(2)
        assert np.allclose(d_optimality.evaluate_jacobian(x), jacobian(x), rtol=1e-3)

    # n_experiment = 1, n_inputs = 2, model: x1**2 + x2**2
    def jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:  # type: ignore
        return -4 * x**3 / (x[0] ** 4 + x[1] ** 4 + delta)

    model = Formula("{x1**2} + {x2**2} - 1")
    d_optimality = DOptimality(
        domain=domain, model=model, n_experiments=n_experiments, delta=1e-3
    )
    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(2)
        assert np.allclose(d_optimality.evaluate_jacobian(x), jacobian(x), rtol=1e-3)

    # n_experiment = 2, n_inputs = 2, model = x1 + x2
    def jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:  # type: ignore
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

    model = Formula("x1 + x2 - 1")
    n_experiments = 2
    d_optimality = DOptimality(
        domain=domain, model=model, n_experiments=n_experiments, delta=1e-3
    )
    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(4)
        assert np.allclose(d_optimality.evaluate_jacobian(x), jacobian(x), rtol=1e-3)

    # n_experiment = 2, n_inputs = 2, model = x1**2 + x2**2
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

    model = Formula("{x1**2} + {x2**2} - 1")
    d_optimality = DOptimality(
        domain=domain, model=model, n_experiments=n_experiments, delta=1e-3
    )

    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(4)
        assert np.allclose(d_optimality.evaluate_jacobian(x), jacobian(x), rtol=1e-3)


def test_DOptimality_evaluate():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    model = get_formula_from_string("linear", domain=domain)

    d_optimality = DOptimality(domain=domain, model=model, n_experiments=3)
    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert np.allclose(d_optimality.evaluate(x), -np.log(4) - np.log(1e-7))


def test_AOptimality_evaluate():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=f"x{i+1}",
                bounds=(0, 1),
            )
            for i in range(3)
        ],
        outputs=[ContinuousOutput(key="y")],
    )
    model = get_formula_from_string("linear", domain=domain)

    a_optimality = AOptimality(domain=domain, model=model, n_experiments=4)

    x = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    assert np.allclose(a_optimality.evaluate(x), 3 * 1.9999991 + 0.9999996)


def test_AOptimality_evaluate_jacobian():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    model = get_formula_from_string("linear", domain=domain)

    a_optimality = AOptimality(domain=domain, model=model, n_experiments=2, delta=0)

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
    model = get_formula_from_string("linear", domain=domain)

    e_optimality = EOptimality(domain=domain, model=model, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    min_eigval = 0.5 * (
        x[0] ** 2
        - np.sqrt(
            x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + 8 * x[0] * x[1] + x[1] ** 4 + 4
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
    model = get_formula_from_string("linear", domain=domain)

    e_optimality = EOptimality(domain=domain, model=model, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    def grad(x):
        temp = np.sqrt(
            x[0] ** 4 + 2 * x[0] ** 2 * x[1] ** 2 + 8 * x[0] * x[1] + x[1] ** 4 + 4
        )
        return np.array(
            [
                (x[0] ** 3 + x[0] * x[1] ** 2 + 2 * x[1]) / temp - x[0],
                (x[1] ** 3 + x[1] * x[0] ** 2 + 2 * x[0]) / temp - x[1],
            ]
        )

    assert np.allclose(e_optimality.evaluate_jacobian(x), grad(x))


def test_GOptimality_evaluate():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    model = get_formula_from_string("linear", domain=domain)

    g_optimality = GOptimality(domain=domain, model=model, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    # all eigenvalues are 1 since A = [[1,1],[1,0.5]] is invertible and therefore H = A (A.T A)^-1 A.T = A A^-1 A.T^-1 A.T = 1

    assert np.allclose(g_optimality.evaluate(x), 1)


def test_GOptimality_evaluate_jacobian():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    model = get_formula_from_string("linear", domain=domain)

    g_optimality = GOptimality(domain=domain, model=model, n_experiments=2, delta=0)

    x = np.array([1, 0.5])

    # all eigenvalues are 1 since A = [[1,1],[1,0.5]] is invertible and therefore H = A (A.T A)^-1 A.T = A A^-1 A.T^-1 A.T = 1
    # thus, the jacobian vanishes.

    assert np.allclose(g_optimality.evaluate_jacobian(x), np.zeros(2))


def test_SpaceFilling_evaluate():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    model = get_formula_from_string("linear", domain=domain)

    space_filling = SpaceFilling(domain=domain, model=model, n_experiments=4, delta=0)

    x = np.array([1, 0.6, 0.1, 0.3])

    assert np.allclose(space_filling.evaluate(x), -1.4)


def test_SpaceFilling_evaluate_jacobian():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x1", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    model = get_formula_from_string("linear", domain=domain)

    space_filling = SpaceFilling(domain=domain, model=model, n_experiments=4, delta=0)

    x = np.array([1, 0.4, 0, 0.1])

    assert np.allclose(space_filling.evaluate_jacobian(x), [-1, -1, 2, 0])
