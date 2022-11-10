from abc import abstractmethod
from typing import Dict

import numpy as np
from pydantic.class_validators import root_validator
from pydantic.types import confloat

from bofire.domain.util import BaseModel

# for the return functions we do not distinguish between multiplicative/ additive (i.e. *weight or **weight),
# since when a desirability function is called directly, we only have one objective


class DesirabilityFunction(BaseModel):
    """The base class for all desirability functions"""

    @abstractmethod
    def __call__(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Abstract method to define the call function for the class DesirabilityFunction

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
        """Generate serialized version of the desirability function.

        Returns:
            Dict: Serialized version of the desirability function as dictionary.
        """
        return {
            "type": self.__class__.__name__,
            **self.dict(),
        }

    @staticmethod
    def from_config(config: Dict) -> "DesirabilityFunction":
        """Generate desirability function out of serialized version.

        Args:
            config (Dict): Serialized version of a desirability function

        Returns:
            DesirabilityFunction: Instaniated desirability function of the type specified in the `config`.
        """
        mapper = {
            "MaxIdentityDesirabilityFunction": MaxIdentityDesirabilityFunction,
            "MinIdentityDesirabilityFunction": MinIdentityDesirabilityFunction,
            "DeltaIdentityDesirabilityFunction": DeltaIdentityDesirabilityFunction,
            "MaxSigmoidDesirabilityFunction": MaxSigmoidDesirabilityFunction,
            "MinSigmoidDesirabilityFunction": MinSigmoidDesirabilityFunction,
            "ConstantDesirabilityFunction": ConstantDesirabilityFunction,
            "TargetDesirabilityFunction": TargetDesirabilityFunction,
            "CloseToTargetDesirabilityFunction": CloseToTargetDesirabilityFunction,
        }
        return mapper[config["type"]](**config)


class IdentityDesirabilityFunction(DesirabilityFunction):
    """A desirability function returning the identity as reward.
    The return can be scaled, when a lower and upper bound are provided.

    Attributes:
        w (float): float between zero and one for weighting the desirability function
        lower_bound (float, optional): Lower bound for normalizing the desirability function between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the desirability function between zero and one. Defaults to one.
    """

    w: confloat(gt=0, le=1)
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


class MaxIdentityDesirabilityFunction(IdentityDesirabilityFunction):
    """Child class from the identity function without modifications, since the parent class is already defined as maximization

    Attributes:
        w (float): float between zero and one for weighting the desirability function
        lower_bound (float, optional): Lower bound for normalizing the desirability function between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the desirability function between zero and one. Defaults to one.
    """

    pass


class MinIdentityDesirabilityFunction(IdentityDesirabilityFunction):
    """Class returning the negative identity as reward.

    Attributes:
        w (float): float between zero and one for weighting the desirability function
        lower_bound (float, optional): Lower bound for normalizing the desirability function between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the desirability function between zero and one. Defaults to one.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """The call function returning a reward for passed x values

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: The negative identity as reward, might be normalized to the passed lower and upper bounds
        """
        return -1.0 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound)


class DeltaIdentityDesirabilityFunction(IdentityDesirabilityFunction):
    """Class returning the difference between a reference value and identity as reward

    Attributes:
        w (float): float between zero and one for weighting the desirability function
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


class SigmoidDesirabilityFunction(DesirabilityFunction):
    """Base class for all sigmoid shaped desirability functions

    Attributes:
        w (float): float between zero and one for weighting the desirability function.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Turning point of the sigmoid function.
    """

    steepness: confloat(gt=0)
    tp: float
    w: confloat(gt=0, le=1)


class MaxSigmoidDesirabilityFunction(SigmoidDesirabilityFunction):
    """Class for a maximizing sigmoid desirability function

    Attributes:
        w (float): float between zero and one for weighting the desirability function.
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


class MinSigmoidDesirabilityFunction(SigmoidDesirabilityFunction):
    """Class for a minimizing a sigmoid desirability function

    Attributes:
        w (float): float between zero and one for weighting the desirability function.
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


class ConstantDesirabilityFunction(DesirabilityFunction):
    """Constant desirability function to allow constrained output features which should not be optimized

    Attributes:
        w (float): float between zero and one for weighting the desirability function.
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


class AbstractTargetDesirabilityFunction(DesirabilityFunction):
    w: confloat(gt=0, le=1)
    target_value: float
    tolerance: confloat(ge=0)

    def plot_details(self, ax):
        """Plot function highlighting the tolerance area of the desirability function

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


class CloseToTargetDesirabilityFunction(AbstractTargetDesirabilityFunction):
    exponent: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (
            x - self.target
        ).abs() ** self.exponent - self.tolerance**self.exponent


class TargetDesirabilityFunction(AbstractTargetDesirabilityFunction):
    """Class for desirability functions for optimizing towards a target value

    Attributes:
        w (float): float between zero and one for weighting the desirability function.
        target_value (float): target value that should be reached.
        tolerance (float): Tolerance for reaching the target. Has to be greater than zero.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.

    """

    steepness: confloat(gt=0)

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
