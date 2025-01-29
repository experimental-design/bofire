import numpy as np
import pandas as pd

from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.constraints.linear import LinearInequalityConstraint
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.continuous import ContinuousInput, ContinuousOutput


def _poly2(x: np.ndarray) -> np.ndarray:
    """Quadratic feature expansion including bias term."""
    return np.concatenate([[1], x, np.outer(x, x)[np.triu_indices(5)]])


class Detergent(Benchmark):
    """Detergent formulation problem.

    There are 5 outputs representing the washing performance on different stain types.
    Each output is modeled as a second degree polynomial.
    The formulation consists of 5 components.
    The sixth input is a filler (water) and is factored out and it's parameter bounds
    0.6 < water < 0.8 result in 2 linear inequality constraints for the other parameters.
    """

    def __init__(self):
        super().__init__()
        # coefficients for the 2-order polynomial; generated with
        # base = 3 * np.ones((1, 5))
        # scale = PolynomialFeatures(degree=2).fit_transform(base).T
        # coef = np.random.RandomState(42).normal(scale=scale, size=(len(scale), 5))
        # coef = np.clip(coef, 0, None)
        self.coef = np.array(
            [
                [0.4967, 0.0, 0.6477, 1.523, 0.0],
                [0.0, 4.7376, 2.3023, 0.0, 1.6277],
                [0.0, 0.0, 0.7259, 0.0, 0.0],
                [0.0, 0.0, 0.9427, 0.0, 0.0],
                [4.3969, 0.0, 0.2026, 0.0, 0.0],
                [0.3328, 0.0, 1.1271, 0.0, 0.0],
                [0.0, 16.6705, 0.0, 0.0, 7.4029],
                [0.0, 1.8798, 0.0, 0.0, 1.7718],
                [6.6462, 1.5423, 0.0, 0.0, 0.0],
                [0.0, 0.0, 9.5141, 3.0926, 0.0],
                [2.9168, 0.0, 0.0, 5.5051, 9.279],
                [8.3815, 0.0, 0.0, 2.9814, 8.7799],
                [0.0, 0.0, 0.0, 0.0, 7.3127],
                [12.2062, 0.0, 9.0318, 3.2547, 0.0],
                [3.2526, 13.8423, 0.0, 14.0818, 0.0],
                [7.3971, 0.7834, 0.0, 0.8258, 0.0],
                [0.0, 3.214, 13.301, 0.0, 0.0],
                [0.0, 8.2386, 2.9588, 0.0, 4.6194],
                [0.8737, 8.7178, 0.0, 0.0, 0.0],
                [0.0, 2.6651, 2.3495, 0.046, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        self._domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=[0.0, 0.2]),
                ContinuousInput(key="x2", bounds=[0.0, 0.3]),
                ContinuousInput(key="x3", bounds=[0.02, 0.2]),
                ContinuousInput(key="x4", bounds=[0.0, 0.06]),
                ContinuousInput(key="x5", bounds=[0.0, 0.04]),
            ],
            outputs=[ContinuousOutput(key=f"y{i+1}") for i in range(5)],
            constraints=[
                LinearInequalityConstraint(
                    features=["x1", "x2", "x3", "x4", "x5"],
                    coefficients=[-1] * 5,
                    rhs=-0.2,
                ),
                LinearInequalityConstraint(
                    features=["x1", "x2", "x3", "x4", "x5"],
                    coefficients=[1] * 5,
                    rhs=0.4,
                ),
            ],
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        x = np.atleast_2d(X[self.domain.inputs.get_keys()])
        xp = np.stack([_poly2(xi) for xi in x], axis=0)
        return pd.DataFrame(
            xp @ self.coef,
            columns=self.domain.outputs.get_keys(),
            index=X.index,
        )
