import warnings
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from pydantic import PositiveFloat

from bofire.data_models.constraints.api import (
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Inputs
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.strategies.api import ShortestPathStrategy as DataModel
from bofire.strategies.strategy import Strategy, make_strategy


class ShortestPathStrategy(Strategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        """Initialize the ShortestPath strategy.

        Args:
            data_model (DataModel): The data model of the shortest path strategy.

        """
        self.start = pd.Series(data_model.start)
        self.end = pd.Series(data_model.end)
        self.atol = data_model.atol
        super().__init__(data_model=data_model, **kwargs)

    @property
    def continuous_inputs(self) -> Inputs:
        """Returns the continuous inputs from the domain.

        Returns:
            Inputs: The continuous inputs from the domain.

        """
        return self.domain.inputs.get(ContinuousInput)

    def get_linear_constraints(
        self,
        constraints: Constraints,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the linear constraints in the form of matrices A and b, where Ax = b for
            equality constraints and Ax <= b for inequality constraints.

        Args:
            constraints (Constraints): The `Constraints` object containing the linear constraints.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the matrices A and b.

        """
        inputs = self.continuous_inputs
        keys = inputs.get_keys()
        b = np.array([c.rhs for c in constraints])
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
        """Takes a starting point and returns the next step in the shortest path.

        Args:
            start (pd.Series): The starting point for the shortest path.

        Returns:
            pd.Series: The next step in the shortest path.

        """
        inputs = self.continuous_inputs
        lower, upper = inputs.get_bounds(
            specs={},
            reference_experiment=start[inputs.get_keys()],
        )
        x = cp.Variable(len(inputs))
        cost = cp.sum_squares(x - self.end[inputs.get_keys()])
        constraints = [
            np.eye(len(inputs)) @ x >= np.array(lower),
            np.eye(len(inputs)) @ x <= np.array(upper),
        ]
        if len(self.domain.constraints.get(LinearEqualityConstraint)) > 0:
            A, b = self.get_linear_constraints(
                self.domain.constraints.get(LinearEqualityConstraint),
            )
            constraints.append(A @ x == b)
        if len(self.domain.constraints.get(LinearInequalityConstraint)) > 0:
            A, b = self.get_linear_constraints(
                self.domain.constraints.get(LinearInequalityConstraint),
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
        """Perform the shortest path strategy to determine the optimal path from the start point to the end point.

        Args:
            candidate_count (Optional[int]): The number of candidates to propose. This argument is ignored by the ShortestPath
                strategy as it automatically determines how many candidates to propose.

        Returns:
            pd.DataFrame: A DataFrame containing the steps taken during the shortest path strategy. Each row represents a step
                and each column represents a feature.

        Raises:
            ValueError: If `candidate_count` is not None, as the ShortestPath strategy ignores the specified value and
                automatically determines how many candidates to propose.

        """
        if candidate_count is not None:
            warnings.warn(
                "ShortestPathStrategy will ignore the specified value of candidate_count. "
                "The strategy automatically determines how many candidates to "
                "propose.",
                UserWarning,
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
        """Checks if there are sufficient experiments available.

        Returns:
            bool: True if there are sufficient experiments, False otherwise.

        """
        return True

    @classmethod
    def make(
        cls,
        domain: Domain,
        start: Dict[str, float | str] | None = None,
        end: Dict[str, float | str] | None = None,
        atol: PositiveFloat | None = None,
        seed: int | None = None,
    ):
        """Represents a strategy for finding the shortest path between two points
        Args:
            start: The starting point of the path.
            end: The ending point of the path.
            atol: The absolute tolerance used for numerical comparisons."""
        return make_strategy(cls, DataModel, locals())
