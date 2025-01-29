from typing import Literal, Optional, Union

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

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values
            x_adapt (np.ndarray): An array of x values which are used to update the objective parameters on the fly.

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.

        """
        return 1 / (1 + np.exp(-1 * self.steepness * (x - self.tp)))


class MovingMaximizeSigmoidObjective(SigmoidObjective):
    """Class for a maximizing sigmoid objective with a moving turning point that depends on so far observed x values.

    Attributes:
        w (float): float between zero and one for weighting the objective when used in a weighting based strategy.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Relative turning point of the sigmoid function. The actual turning point is calculated by adding
            the maximum of the observed x values to the relative turning point.

    """

    type: Literal["MovingMaximizeSigmoidObjective"] = "MovingMaximizeSigmoidObjective"

    def get_adjusted_tp(self, x: Union[pd.Series, np.ndarray]) -> float:
        """Get the adjusted turning point for the sigmoid function.

        Args:
            x (np.ndarray): An array of x values

        Returns:
            float: The adjusted turning point for the sigmoid function.

        """
        return x.max() + self.tp

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Union[pd.Series, np.ndarray],
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values
            x_adapt (np.ndarray): An array of x values which are used to update the objective parameters on the fly.

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.

        """
        return 1 / (
            1 + np.exp(-1 * self.steepness * (x - self.get_adjusted_tp(x_adapt)))
        )


class MinimizeSigmoidObjective(SigmoidObjective):
    """Class for a minimizing a sigmoid objective

    Attributes:
        w (float): float between zero and one for weighting the objective.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Turning point of the sigmoid function.

    """

    type: Literal["MinimizeSigmoidObjective"] = "MinimizeSigmoidObjective"

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values
            x_adapt (np.ndarray): An array of x values which are used to update the objective parameters on the fly.

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.

        """
        return 1 - 1 / (1 + np.exp(-1 * self.steepness * (x - self.tp)))
