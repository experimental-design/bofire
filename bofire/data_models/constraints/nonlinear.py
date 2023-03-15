from typing import Literal

import numpy as np
import pandas as pd

from bofire.data_models.constraints.constraint import Constraint


class NonlinearConstraint(Constraint):
    """Base class for nonlinear equality and inequality constraints.

    Attributes:
        expression: Mathematical expression that can be evaluated by `pandas.eval`.
    """

    expression: str

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        return experiments.eval(self.expression)


class NonlinearEqualityConstraint(NonlinearConstraint):
    """Nonlinear inequality constraint of the form 'expression <= 0'.

    Attributes:
        expression: Mathematical expression that can be evaluated by `pandas.eval`.
    """

    type: Literal["NonlinearEqualityConstraint"] = "NonlinearEqualityConstraint"

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        return pd.Series(np.isclose(self(experiments), 0), index=experiments.index)

    def __str__(self):
        return f"{self.expression}==0"


class NonlinearInequalityConstraint(NonlinearConstraint):
    """Linear inequality constraint of the form 'expression == 0'.

    Attributes:
        expression: Mathematical expression that can be evaluated by `pandas.eval`.
    """

    type: Literal["NonlinearInequalityConstraint"] = "NonlinearInequalityConstraint"

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        # we allow here for numerical noise
        noise = 10e-6
        return self(experiments) <= 0 + noise

    def __str__(self):
        return f"{self.expression}<=0"
