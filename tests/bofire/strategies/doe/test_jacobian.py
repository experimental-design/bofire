import numpy as np
import pandas as pd
import pytest
from formulaic import Formula

from bofire.domain import Domain
from bofire.domain.features import ContinuousInput, ContinuousOutput
from bofire.strategies.doe.jacobian import (
    JacobianForLogdet,
    default_jacobian_building_block,
)


def test_default_jacobian_building_block():
    # "small" model

    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(3)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    vars = domain.inputs.get_keys()
    model_terms = np.array(Formula("x1 + x2 + x3 + x1:x2 + {x3**2}").terms, dtype=str)
    x = [1, 2, 3]

    jacobian_building_block = default_jacobian_building_block(vars, model_terms)

    B = np.zeros(shape=(3, 6))
    B[:, 1:4] = np.eye(3)
    B[:, 4] = np.array([0, 0, 6])
    B[:, 5] = np.array([2, 1, 0])

    assert np.allclose(B, jacobian_building_block(x))

    # fully quadratic model
    model_terms = np.array(
        Formula(
            "x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + {x1**2} + {x2**2} + {x3**2}"
        ).terms,
        dtype=str,
    )
    x = [1, 2, 3]

    jacobian_building_block = default_jacobian_building_block(vars, model_terms)

    B = np.zeros(shape=(3, 10))
    B[:, 1:4] = np.eye(3)
    B[:, 4:7] = 2 * np.diag(x)
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

    assert np.allclose(B, jacobian_building_block(x))

    # unsupported model
    model_terms = np.array(Formula("{x1**4} - 1").terms, dtype=str)
    x = [1, 2, 3]

    jacobian_building_block = default_jacobian_building_block(vars, model_terms)

    with pytest.raises(KeyError):
        jacobian_building_block(x)

    # fully cubic model
    vars = ["x1", "x2", "x3", "x4", "x5"]
    n_vars = 5

    formula = ""
    for name in vars:
        formula += name + " + "

    for name in vars:
        formula += "{" + name + "**2} + "
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            term = str(Formula(vars[j] + ":" + vars[i] + "-1").terms[0]) + " + "
            formula += term

    for name in vars:
        formula += "{" + name + "**3} + "
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            for k in range(j + 1, n_vars):
                term = (
                    str(
                        Formula(vars[k] + ":" + vars[j] + ":" + vars[i] + "-1").terms[0]
                    )
                    + " + "
                )
                formula += term
    formula = Formula(formula[:-3])
    model_terms = np.array(formula.terms, dtype=str)
    x = [1, 2, 3, 4, 5]
    jacobian_building_block = default_jacobian_building_block(vars, model_terms)

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

    assert np.allclose(B, jacobian_building_block(x))


def test_JacobianForLogdet_instantiation():
    # default jacobian building block
    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(3)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    model = Formula("x1 + x2 + x3 + x1:x2 + {x3**2}")
    n_experiments = 2

    J = JacobianForLogdet(domain, model, n_experiments)

    assert isinstance(J.domain, Domain)
    assert all(np.array(J.domain.inputs.get_keys()) == np.array(["x1", "x2", "x3"]))
    for i in J.domain.inputs.get():
        assert isinstance(i, ContinuousInput)
        assert i.upper_bound == 1
        assert i.lower_bound == 0
    assert all(np.array(J.domain.outputs.get_keys()) == np.array(["y"]))

    assert isinstance(J.model, Formula)
    assert all(J.model.terms == np.array(["1", "x1", "x2", "x3", "x3**2", "x1:x2"]))

    x = [1, 2, 3]
    B = np.zeros(shape=(3, 6))
    B[:, 1:4] = np.eye(3)
    B[:, 4] = np.array([0, 0, 6])
    B[:, 5] = np.array([2, 1, 0])

    assert np.allclose(B, J.jacobian_building_block(x))
    assert np.shape(J.jacobian(np.array([[1, 1, 1], [2, 2, 2]]))) == (6,)

    # custom jacobian building block: 5th order model
    def custom_jacobian_building_block(x: np.ndarray) -> np.ndarray:
        x = np.array(x)
        B = np.zeros(shape=(3, 4))
        B[:, 1:] = 5 * np.diag(x**4)
        return B

    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(3)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    model = Formula("{x1**5} + {x2**5} + {x3**5}")
    n_experiments = 3

    J = JacobianForLogdet(domain, model, n_experiments, custom_jacobian_building_block)

    x = np.array([1, 2, 3])
    B = np.zeros(shape=(3, 4))
    B[:, 1:] = 5 * np.diag(x**4)

    assert np.allclose(B, J.jacobian_building_block(x))
    assert np.shape(J.jacobian(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))) == (9,)


def test_JacobianForLogdet_jacobian():
    # n_experiment = 1, n_inputs = 2, model: x1 + x2
    def jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:  # type: ignore
        return -2 * x / (x[0] ** 2 + x[1] ** 2 + delta)

    domain = Domain(
        input_features=[
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(2)
        ],
        output_features=[ContinuousOutput(key="y")],
    )

    model = Formula("x1 + x2 - 1")
    n_experiments = 1
    J = JacobianForLogdet(domain, model, n_experiments, delta=1e-3)

    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(2)
        assert np.allclose(J.jacobian(x), jacobian(x), rtol=1e-3)

    # n_experiment = 1, n_inputs = 2, model: x1**2 + x2**2
    def jacobian(x: np.ndarray, delta=1e-3) -> np.ndarray:  # type: ignore
        return -4 * x**3 / (x[0] ** 4 + x[1] ** 4 + delta)

    model = Formula("{x1**2} + {x2**2} - 1")
    J = JacobianForLogdet(domain, model, n_experiments, delta=1e-3)

    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(2)
        assert np.allclose(J.jacobian(x), jacobian(x), rtol=1e-3)

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
    J = JacobianForLogdet(domain, model, n_experiments, delta=1e-3)

    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(4)
        assert np.allclose(J.jacobian(x), jacobian(x), rtol=1e-3)

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
    J = JacobianForLogdet(domain, model, n_experiments, delta=1e-3)

    np.random.seed(1)
    for _ in range(10):
        x = np.random.rand(4)
        assert np.allclose(J.jacobian(x), jacobian(x), rtol=1e-3)
