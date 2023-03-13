from typing import Callable, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from torch import Tensor

from bofire.data_models.objectives.objective import (
    BotorchConstrainedObjective,
    Objective,
    TGe0,
    TGt0,
    TWeight,
)


class CloseToTargetObjective(Objective):
    # TODO: add docstring to CloseToTargetObjective

    type: Literal["CloseToTargetObjective"] = "CloseToTargetObjective"
    w: TWeight = 1
    target_value: float
    exponent: float

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        return -1 * (np.abs(x - self.target_value) ** self.exponent)


class TargetObjective(Objective, BotorchConstrainedObjective):
    """Class for objectives for optimizing towards a target value

    Attributes:
        w (float): float between zero and one for weighting the objective.
        target_value (float): target value that should be reached.
        tolerance (float): Tolerance for reaching the target. Has to be greater than zero.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.

    """

    type: Literal["TargetObjective"] = "TargetObjective"
    w: TWeight = 1
    target_value: float
    tolerance: TGe0
    steepness: TGt0

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """The call function returning a reward for passed x values.

        Args:
            x (np.array): An array of x values

        Returns:
            np.array: An array of reward values calculated by the product of two sigmoidal shaped functions resulting in a maximum at the target value.
        """
        return (
            1
            / (
                1
                + np.exp(
                    -1 * self.steepness * (x - (self.target_value - self.tolerance))
                )
            )
            * (
                1
                - 1
                / (
                    1.0
                    + np.exp(
                        -1 * self.steepness * (x - (self.target_value + self.tolerance))
                    )
                )
            )
        )

    def plot_details(self, ax):
        """Plot function highlighting the tolerance area of the objective

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes object

        Returns:
            matplotlib.axes.Axes: The object to be plotted
        """
        ax.axvline(self.target_value, color="black")
        ax.axvspan(
            self.target_value - self.tolerance,
            self.target_value + self.tolerance,
            color="gray",
            alpha=0.5,
        )
        return ax

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
        return [
            lambda Z: (Z[..., idx] - (self.target_value - self.tolerance)) * -1.0,
            lambda Z: (Z[..., idx] - (self.target_value + self.tolerance)),
        ], [1.0 / self.steepness, 1.0 / self.steepness]
