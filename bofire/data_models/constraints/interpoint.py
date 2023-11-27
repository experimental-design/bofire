import math
from typing import Annotated, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.constraints.constraint import Constraint


class InterpointConstraint(Constraint):
    """An interpoint constraint describes required relationships between individual
    candidates when asking a strategy for returning more than one candidate.
    """

    type: str


class InterpointEqualityConstraint(InterpointConstraint):
    """Constraint that forces that values of a certain feature of a set/batch of
    candidates should have the same value.

    Attributes:
        feature(str): The constrained feature.
        multiplicity(int): The multiplicity of the constraint, stating how many
            values of the feature in the batch should have always the same value.
    """

    type: Literal["InterpointEqualityConstraint"] = "InterpointEqualityConstraint"
    feature: str
    multiplicity: Optional[Annotated[int, Field(ge=2)]]

    def is_fulfilled(
        self, experiments: pd.DataFrame, tol: Optional[float] = 1e-6
    ) -> pd.Series:
        multiplicity = self.multiplicity or len(experiments)
        for i in range(math.ceil(len(experiments) / multiplicity)):
            batch = experiments[self.feature].values[
                i * multiplicity : min((i + 1) * multiplicity, len(experiments))
            ]
            if not np.allclose(batch, batch[0]):
                return pd.Series([False])
        return pd.Series([True])

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Method `__call__` currently not implemented.")

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Method `jacobian` currently not implemented.")
