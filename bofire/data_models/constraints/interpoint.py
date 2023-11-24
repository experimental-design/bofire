import math
from abc import abstractmethod
from typing import Annotated, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.constraints.constraint import Constraint


class InterpointConstraint(Constraint):
    type: str

    @abstractmethod
    def is_fulfilled(
        self, experiments: pd.DataFrame, tol: Optional[float] = 1e-6
    ) -> bool:
        """Abstract method to check if a constraint is fulfilled for the whole set of experiments.

        Args:
            experiments (pd.DataFrame): Dataframe to check constraint fulfillment.
            tol (float, optional): tolerance parameter. A constraint is considered as not fulfilled if
                the violation is larger than tol. Defaults to 1e-6.

        Returns:
            bool: True if fulfilled else False
        """
        pass


class InterpointEqualityConstraint(InterpointConstraint):
    type: Literal["InterpointEqualityConstraint"] = "InterpointEqualityConstraint"
    feature: str
    multiplicity: Optional[Annotated[int, Field(ge=2)]]

    def is_fulfilled(
        self, experiments: pd.DataFrame, tol: Optional[float] = 1e-6
    ) -> bool:
        multiplicity = self.multiplicity or len(experiments)
        for i in range(math.floor(len(experiments) / multiplicity)):
            batch = experiments[self.feature].values[
                i * multiplicity : (i + 1) * multiplicity
            ]
            if not np.allclose(batch, batch[0]):
                return False
        if len(experiments) % multiplicity > 0:
            batch = experiments[self.feature].values[
                -(len(experiments) % multiplicity) :
            ]
            print(batch)
            return np.allclose(batch, batch[0])
        return True
