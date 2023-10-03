from typing import List, Literal, Union

import numpy as np
import pandas as pd

from bofire.data_models.objectives.objective import (
    ConstrainedObjective,
    Objective,
    TWeight,
)


class CategoricalObjective(Objective, ConstrainedObjective):
    """Compute the categorical objective value as:

        Po where P is an [n, c] matrix where each row is a probability vector
        (P[i, :].sum()=1 for all i) and o is a column vector of objective values

    Attributes:
        w (float): float between zero and one for weighting the objective.
        weights (list): list of values of size c (c is number of categories) such that the i-th entry is in (0, 1)
    """

    w: TWeight = 1.0
    weights: List[float]
    eta: float = 1.0
    categories: Union[List[str], None] = None

    type: Literal["CategoricalObjective"] = "CategoricalObjective"

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """The call function returning a probabilistic reward for x.

        Args:
            x (np.ndarray): A matrix of x values

        Returns:
            np.ndarray: A reward calculated as inner product of probabilities and feasible objectives.
        """
        return x.map(dict(zip(self.categories, self.weights)))
