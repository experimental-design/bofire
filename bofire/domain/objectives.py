from __future__ import annotations

from abc import abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from pydantic.class_validators import root_validator
from pydantic.types import confloat

from bofire.domain.util import BaseModel

# for the return functions we do not distinguish between multiplicative/ additive (i.e. *weight or **weight),
# since when a objective is called directly, we only have one objective

TGt0 = confloat(gt=0)
TGe0 = confloat(ge=0)
TWeight = confloat(gt=0, le=1)


class Objective(BaseModel):
    """The base class for all objectives"""

    @abstractmethod
    def __call__(self, x: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to define the call function for the class Objective

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: The desirability of the passed x values
        """
        pass

    def plot_details(self, ax):
        """
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes object

        Returns:
            matplotlib.axes.Axes: The object to be plotted
        """
        return ax

    def to_config(self) -> Dict:
        """Generate serialized version of the objective.

        Returns:
            Dict: Serialized version of the objective as dictionary.
        """
        return {
            "type": self.__class__.__name__,
            **self.dict(),
        }

    @staticmethod
    def from_config(config: Dict) -> "Objective":
        """Generate objective out of serialized version.

        Args:
            config (Dict): Serialized version of an objective.

        Returns:
            Objective: Instaniated objective of the type specified in the `config`.
        """
        mapper = {
            "MaximizeObjective": MaximizeObjective,
            "MinimizeObjective": MinimizeObjective,
            "DeltaObjective": DeltaObjective,
            "MaximizeSigmoidObjective": MaximizeSigmoidObjective,
            "MinimizeSigmoidObjective": MinimizeSigmoidObjective,
            "ConstantObjective": ConstantObjective,
            "TargetObjective": TargetObjective,
            "CloseToTargetObjective": CloseToTargetObjective,
        }
        return mapper[config["type"]](**config)


class IdentityObjective(Objective):
    """An objective returning the identity as reward.
    The return can be scaled, when a lower and upper bound are provided.

    Attributes:
        w (float): float between zero and one for weighting the objective
        lower_bound (float, optional): Lower bound for normalizing the objective between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the objective between zero and one. Defaults to one.
    """

    w: TWeight
    lower_bound: float = 0
    upper_bound: float = 1

    @root_validator(pre=False)
    def validate_lower_upper(cls, values):
        """Validation function to ensure that lower bound is always greater the upper bound

        Args:
            values (Dict): The attributes of the class

        Raises:
            ValueError: when a lower bound higher than the upper bound is passed

        Returns:
            Dict: The attributes of the class
        """
        if values["lower_bound"] > values["upper_bound"]:
            raise ValueError(
                f'lower bound must be <= upper bound, got {values["lower_bound"]} > {values["upper_bound"]}'
            )
        return values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The call function returning a reward for passed x values

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: The identity as reward, might be normalized to the passed lower and upper bounds
        """
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)


class MaximizeObjective(IdentityObjective):
    """Child class from the identity function without modifications, since the parent class is already defined as maximization

    Attributes:
        w (float): float between zero and one for weighting the objective
        lower_bound (float, optional): Lower bound for normalizing the objective between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the objective between zero and one. Defaults to one.
    """

    pass


class MinimizeObjective(IdentityObjective):
    """Class returning the negative identity as reward.

    Attributes:
        w (float): float between zero and one for weighting the objective
        lower_bound (float, optional): Lower bound for normalizing the objective between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the objective between zero and one. Defaults to one.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The call function returning a reward for passed x values

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: The negative identity as reward, might be normalized to the passed lower and upper bounds
        """
        return -1.0 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound)


class DeltaObjective(IdentityObjective):
    """Class returning the difference between a reference value and identity as reward

    Attributes:
        w (float): float between zero and one for weighting the objective
        ref_point (float): Reference value.
        scale (float, optional): Scaling factor for the difference. Defaults to one.
    """

    ref_point: float
    scale: float = 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The call function returning a reward for passed x values

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: The difference between reference and the x value as reward, might be scaled with a passed scaling value
        """
        return (self.ref_point - x) * self.scale


class SigmoidObjective(Objective):
    """Base class for all sigmoid shaped objectives

    Attributes:
        w (float): float between zero and one for weighting the objective.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Turning point of the sigmoid function.
    """

    steepness: TGt0
    tp: float
    w: TWeight


class MaximizeSigmoidObjective(SigmoidObjective):
    """Class for a maximizing sigmoid objective

    Attributes:
        w (float): float between zero and one for weighting the objective.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Turning point of the sigmoid function.

    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.
        """
        return 1 - 1 / (1 + np.exp(-1 * self.steepness * (x - self.tp)))


class ConstantObjective(Objective):
    """Constant objective to allow constrained output features which should not be optimized

    Attributes:
        w (float): float between zero and one for weighting the objective.
    """

    w: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The call function returning the fixed value as reward

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: An array of passed constants with the shape of the passed x values array.
        """
        return np.ones(x.shape) * self.w


class AbstractTargetObjective(Objective):
    w: TWeight
    target_value: float
    tolerance: TGe0

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


class CloseToTargetObjective(AbstractTargetObjective):
    exponent: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (
            np.abs(x - self.target_value) ** self.exponent
            - self.tolerance**self.exponent
        )


class TargetObjective(AbstractTargetObjective):
    """Class for objectives for optimizing towards a target value

    Attributes:
        w (float): float between zero and one for weighting the objective.
        target_value (float): target value that should be reached.
        tolerance (float): Tolerance for reaching the target. Has to be greater than zero.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.

    """

    steepness: TGt0

    def __call__(self, x: np.ndarray) -> np.ndarray:
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
