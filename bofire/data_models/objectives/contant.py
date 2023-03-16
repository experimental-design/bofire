from typing import Literal, Union

import numpy as np
import pandas as pd

from bofire.data_models.objectives.objective import Objective, TWeight


class ConstantObjective(Objective):
    """Constant objective to allow constrained output features which should not be optimized

    Attributes:
        w (float): float between zero and one for weighting the objective.
        value (float): constant return value
    """

    type: Literal["ConstantObjective"] = "ConstantObjective"
    w: TWeight = 1
    value: float

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """The call function returning the fixed value as reward

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: An array of passed constants with the shape of the passed x values array.
        """
        return np.ones(x.shape) * self.value
