from typing import Literal

import numpy as np
import pandas as pd

from bofire.data_models.constraints.constraint import Constraint


class NonlinearConstraint(Constraint):
    # TODO: add docstring to NonLinearConstraint

    type: Literal["NonlinearConstraint"] = "NonlinearConstraint"
    expression: str

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        return experiments.eval(self.expression)


class NonlinearEqualityConstraint(NonlinearConstraint):
    # TODO: add docstring to NonlinearEqualityConstraint
    type: Literal["NonlinearEqualityConstraint"] = "NonlinearEqualityConstraint"

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        return pd.Series(np.isclose(self(experiments), 0), index=experiments.index)

    def __str__(self):
        return f"{self.expression}==0"


class NonlinearInequalityConstraint(NonlinearConstraint):
    # TODO: add docstring to NonlinearInequalityConstraint
    type: Literal["NonlinearInequalityConstraint"] = "NonlinearInequalityConstraint"

    def is_fulfilled(self, experiments: pd.DataFrame) -> pd.Series:
        # we allow here for numerical noise
        noise = 10e-6
        return self(experiments) <= 0 + noise

    def __str__(self):
        return f"{self.expression}<=0"
