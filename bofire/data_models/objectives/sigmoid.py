from typing import Literal, Union

import numpy as np
import pandas as pd

from bofire.data_models.objectives.objective import (
    ConstrainedObjective,
    Objective,
    TGt0,
    TWeight,
)


class SigmoidObjective(Objective, ConstrainedObjective):
    """Base class for all sigmoid shaped objectives

    Attributes:
        w (float): float between zero and one for weighting the objective.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Turning point of the sigmoid function.
    """

    steepness: TGt0
    tp: float
    w: TWeight = 1


class MaximizeSigmoidObjective(SigmoidObjective):
    """Class for a maximizing sigmoid objective

    Attributes:
        w (float): float between zero and one for weighting the objective.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Turning point of the sigmoid function.

    """

    type: Literal["MaximizeSigmoidObjective"] = "MaximizeSigmoidObjective"

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.
        """
        return 1 / (1 + np.exp(-1 * self.steepness * (x - self.tp)))


class MinimizeSigmoidObjective(SigmoidObjective):
    """Class for a minimizing a sigmoid objective

    Attributes:
        w (float): float between zero and one for weighting the objective.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Turning point of the sigmoid function.
    """

    type: Literal["MinimizeSigmoidObjective"] = "MinimizeSigmoidObjective"

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.
        """
        return 1 - 1 / (1 + np.exp(-1 * self.steepness * (x - self.tp)))
