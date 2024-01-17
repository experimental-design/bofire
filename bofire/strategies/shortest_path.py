from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import (
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Inputs
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.strategies.api import ShortestPathStrategy as DataModel
from bofire.strategies.strategy import Strategy


class ShortestPathStrategy(Strategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.start = pd.Series(data_model.start)
        self.end = pd.Series(data_model.end)
        self.atol = data_model.atol
        super().__init__(data_model=data_model, **kwargs)

    @property
    def continuous_inputs(self) -> Inputs:
        return self.domain.inputs.get(ContinuousInput)  # type: ignore

    def get_linear_constraints(
        self, constraints: Constraints
    ) -> Tuple[np.ndarray, np.ndarray]:
        inputs = self.continuous_inputs
        keys = inputs.get_keys()
        b = np.array([c.rhs for c in constraints])  # type: ignore
        A = np.zeros([len(constraints), len(inputs)])
        for i, c in enumerate(constraints):
            assert isinstance(c, LinearConstraint)
            for key, coef in zip(c.features, c.coefficients):
                feat = inputs.get_by_key(key)
                assert isinstance(feat, ContinuousInput)
                if feat.is_fixed():
                    b[i] -= feat.fixed_value()[0] * coef  # type: ignore
                else:
                    A[i, keys.index(key)] = coef
        return A, b

    def step(self, start: pd.Series) -> pd.Series:
        inputs = self.continuous_inputs
        lower, upper = inputs.get_bounds(
            specs={}, reference_experiment=start[inputs.get_keys()]
        )
        x = cp.Variable(len(inputs))
        cost = cp.sum_squares(x - self.end[inputs.get_keys()])
        constraints = [
            np.eye(len(inputs)) @ x >= np.array(lower),
            np.eye(len(inputs)) @ x <= np.array(upper),
        ]
        if len(self.domain.constraints.get(LinearEqualityConstraint)) > 0:
            A, b = self.get_linear_constraints(
                self.domain.constraints.get(LinearEqualityConstraint)
            )
            constraints.append(A @ x == b)
        if len(self.domain.constraints.get(LinearInequalityConstraint)) > 0:
            A, b = self.get_linear_constraints(
                self.domain.constraints.get(LinearInequalityConstraint)
            )
            constraints.append(A @ x <= b)
        prob = cp.Problem(objective=cp.Minimize(cost), constraints=constraints)  # type: ignore
        prob.solve(solver=cp.CLARABEL)
        step = pd.Series(index=inputs.get_keys(), data=x.value)
        # add other features based on start
        for key in self.domain.inputs.get_keys(excludes=[ContinuousInput], includes=[]):
            step[key] = self.end[key]
        return step

    def _ask(self, candidate_count: Optional[int] = None) -> pd.DataFrame:
        if candidate_count is not None:
            raise ValueError(
                "ShortestPath will ignore the specified value of candidate_count. "
                "The strategy automatically determines how many candidates to "
                "propose."
            )
        start = self.start
        steps = []
        while not np.allclose(
            start[self.continuous_inputs.get_keys()].tolist(),
            self.end[self.continuous_inputs.get_keys()].tolist(),
            atol=self.atol,
        ):
            step = self.step(start=start)
            steps.append(step)
            start = step
        return pd.concat(steps, axis=1).T

    def has_sufficient_experiments(self) -> bool:
        return True
