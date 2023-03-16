from typing import Callable, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from torch import Tensor

from bofire.data_models.objectives.objective import (
    BotorchConstrainedObjective,
    Objective,
    TGt0,
    TWeight,
)


class SigmoidObjective(Objective, BotorchConstrainedObjective):
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

    def to_constraints(
        self, idx: int
    ) -> Tuple[List[Callable[[Tensor], Tensor]], List[float]]:
        """Create a callable that can be used by `botorch.utils.objective.apply_constraints` to setup ouput constrained optimizations.

        Args:
            idx (int): Index of the constraint objective in the list of outputs.

        Returns:
            Tuple[List[Callable[[Tensor], Tensor]], List[float]]: List of callables that can be used by botorch for setting up the constrained objective, and
                list of the corresponding botorch eta values.
        """
        return [lambda Z: (Z[..., idx] - self.tp) * -1.0], [1.0 / self.steepness]


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

    def to_constraints(
        self, idx: int
    ) -> Tuple[List[Callable[[Tensor], Tensor]], List[float]]:
        """Create a callable that can be used by `botorch.utils.objective.apply_constraints` to setup ouput constrained optimizations.

        Args:
            idx (int): Index of the constraint objective in the list of outputs.

        Returns:
            Tuple[List[Callable[[Tensor], Tensor]], List[float]]: List of callables that can be used by botorch for setting up the constrained objective, and
                list of the corresponding botorch eta values.
        """
        return [lambda Z: (Z[..., idx] - self.tp)], [1.0 / self.steepness]
