# warn user if IPOPT scipy interface is not available
try:
    from cyipopt import Problem  # type: ignore
except ImportError:

    class Problem:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "cyipopt is not installed. Install it via `conda install -c conda-forge cyipopt`"
            )


from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.optimize._minimize import LinearConstraint, NonlinearConstraint

from bofire.strategies.doe.objective_base import Objective


class FirstOrderDoEProblem(Problem):  # type: ignore
    def __init__(
        self,
        doe_objective: Objective,
        bounds: List[Tuple[float, float]],
        constraints: Optional[
            List[Union[NonlinearConstraint, LinearConstraint]]
        ] = None,
    ):
        self.doe_objective = doe_objective
        n_vars = doe_objective.n_vars * doe_objective.n_experiments

        # assemble linear constraints to one big linear constraint
        if constraints is None:
            constraints = []
        self.linear_constraints = [
            c for c in constraints if isinstance(c, LinearConstraint)
        ]
        self.A = (
            sparse.coo_array(np.vstack([c.A for c in self.linear_constraints]))
            if len(self.linear_constraints) > 0
            else sparse.coo_matrix([]).T
        )

        # assemble nonlinear constraints to one big nonlinear constraint
        self.nonlinear_constraints = [
            c for c in constraints if isinstance(c, NonlinearConstraint)
        ]

        # generate jacobian structure
        start_row = self.A.shape[0]
        nonlinear_constraint_row = []
        nonlinear_constraint_col = []
        for _ in self.nonlinear_constraints:
            nonlinear_constraint_row.append(
                start_row
                + np.repeat(
                    np.arange(doe_objective.n_experiments), doe_objective.n_vars
                )
            )
            nonlinear_constraint_col.append(np.arange(n_vars))
            start_row += doe_objective.n_experiments
        nonlinear_constraint_row = (
            np.concatenate(nonlinear_constraint_row)
            if len(nonlinear_constraint_row) > 0
            else np.array([])
        )
        nonlinear_constraint_col = (
            np.concatenate(nonlinear_constraint_col)
            if len(nonlinear_constraint_col) > 0
            else np.array([])
        )

        self.row = np.concatenate((self.A.row, nonlinear_constraint_row))
        self.col = np.concatenate((self.A.col, nonlinear_constraint_col))

        self.hessian_structure = np.nonzero(np.ones((4, 4)))

        super().__init__(
            n=int(n_vars),
            m=sum([len(c.lb) for c in constraints]),
            lb=np.array([b[0] for b in bounds]),
            ub=np.array([b[1] for b in bounds]),
            cl=np.array(
                [c.lb for c in self.linear_constraints]
                + [c.lb for c in self.nonlinear_constraints]
            ).flatten(),
            cu=np.array(
                [c.ub for c in self.linear_constraints]
                + [c.ub for c in self.nonlinear_constraints]
            ).flatten(),
        )

    def objective(self, x: np.ndarray) -> float:
        return self.doe_objective.evaluate(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.doe_objective.evaluate_jacobian(x)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        linear = (
            np.array(self.A @ x) if len(self.linear_constraints) > 0 else np.array([])
        )
        if len(linear.shape) == 0:
            linear = np.array([linear])
        nonlinear = (
            np.array([c.fun(x) for c in self.nonlinear_constraints]).flatten()
            if len(self.nonlinear_constraints) > 0
            else np.array([])
        )
        return np.concatenate((linear, nonlinear))

    def jacobian(self, x: np.ndarray):
        linear = self.A.data if len(self.linear_constraints) > 0 else np.array([])
        nonlinear = (
            np.concatenate([c.jac(x, sparse=True) for c in self.nonlinear_constraints])
            if len(self.nonlinear_constraints) > 0
            else np.array([])
        )
        return np.concatenate((linear, nonlinear))

    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.row, self.col


class SecondOrderDoEProblem(FirstOrderDoEProblem):
    def hessian(
        self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float
    ) -> np.ndarray:
        H = obj_factor * np.array(self.doe_objective.evaluate_hessian(x))

        # linear constraints have a vanishing hessian
        for i, c in enumerate(self.nonlinear_constraints):
            H += lagrange[i] * c.hess(x)

        row, col = self.hessianstructure()

        return H[row, col]

    def hessianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.hessian_structure
