"""
ZDT benchmark problem suite.
All problems are bi-objective, have D continuous inputs and are unconstrained.
Zitzler, Deb, Thiele 2000 - Comparison of Multiobjective Evolutionary Algorithms: Empirical Results
http://dx.doi.org/10.1162/106365600568202
"""
import numpy as np
import pandas as pd

from bofire.benchmarks.benchmark import Benchmark
from bofire.domain.domain import Domain
from bofire.domain.features import (
    ContinuousInput,
    ContinuousOutput,
    InputFeatures,
    OutputFeatures,
)
from bofire.domain.objectives import MinimizeObjective


class ZDT1(Benchmark):
    """ZDT-1 benchmark problem."""

    def __init__(self, n_inputs=30):
        self.n_inputs = n_inputs
        input_features = [
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(n_inputs)
        ]
        inputs = InputFeatures(features=input_features)  # type: ignore
        output_features = [
            ContinuousOutput(key=f"y{i+1}", objective=MinimizeObjective(w=1))
            for i in range(2)
        ]
        outputs = OutputFeatures(features=output_features)  # type: ignore
        self._domain = Domain(input_features=inputs, output_features=outputs)

    @property
    def domain(self) -> Domain:
        return self._domain

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self._domain.input_features.get_keys()[1:]].to_numpy()
        g = 1 + 9 / (self.n_inputs - 1) * np.sum(x, axis=1)
        y1 = X["x1"].to_numpy()
        y2 = g * (1 - (y1 / g) ** 0.5)
        return pd.DataFrame({"y1": y1, "y2": y2}, index=X.index)

    def get_optima(self, points=100):
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.sqrt(x)], axis=1)
        return pd.DataFrame(y, columns=self.domain.output_features.get_keys())
