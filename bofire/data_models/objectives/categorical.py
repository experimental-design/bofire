from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import validator

from bofire.data_models.objectives.objective import (
    ConstrainedObjective,
    Objective,
    TWeight,
)


class CategoricalObjective(Objective, ConstrainedObjective):
    """Compute the categorical objective value as:

        Po where P is an [n, c] matrix where each row is a probability vector
        (P[i, :].sum()=1 for all i) and o is a vector of size [c] of objective values

    Attributes:
        w (float): float between zero and one for weighting the objective.
        desirability (tuple): tuple of values of size c (c is number of categories) such that the i-th entry is in (0, 1)
    """

    w: TWeight = 1.0
    desirability: Tuple[float, ...]
    eta: float = 1.0
    type: Literal["CategoricalObjective"] = "CategoricalObjective"

    @validator("desirability")
    def validate_desirability(cls, desirability):
        for w in desirability:
            if w > 1:
                raise ValueError("Objective weight has to be smaller equal than 1.")
            if w < 0:
                raise ValueError("Objective weight has to be larger equal than zero")
        return desirability

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """The call function returning a probabilistic reward for x.

        Args:
            x (np.ndarray): A matrix of x values

        Returns:
            np.ndarray: A reward calculated as inner product of probabilities and feasible objectives.
        """
        print(
            "Categorical objective currently does not have a function. Returning the original input."
        )
        return x
