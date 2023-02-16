from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Callable, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, parse_obj_as
from pydantic.class_validators import root_validator
from torch import Tensor

from bofire.domain.util import PydanticBaseModel

# for the return functions we do not distinguish between multiplicative/ additive (i.e. *weight or **weight),
# since when a objective is called directly, we only have one objective

TGt0 = Annotated[float, Field(type=float, gt=0)]
TGe0 = Annotated[float, Field(type=float, ge=0)]
TWeight = Annotated[float, Field(type=float, gt=0, le=1)]


class BotorchConstrainedObjective:
    """This abstract class offers a convenience routine for transforming sigmoid based objectives to botorch output constraints."""

    @abstractmethod
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
        pass


class Objective(PydanticBaseModel):
    """The base class for all objectives"""

    type: str

    @abstractmethod
    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
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

    @staticmethod
    def from_dict(dict_: dict):
        return parse_obj_as(AnyObjective, dict_)


class IdentityObjective(Objective):
    """An objective returning the identity as reward.
    The return can be scaled, when a lower and upper bound are provided.

    Attributes:
        w (float): float between zero and one for weighting the objective
        lower_bound (float, optional): Lower bound for normalizing the objective between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the objective between zero and one. Defaults to one.
    """

    type: Literal["IdentityObjective"] = "IdentityObjective"
    w: TWeight = 1
    lower_bound: float = 0
    upper_bound: float = 1

    @root_validator(pre=False, skip_on_failure=True)
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

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
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

    type: Literal["MaximizeObjective"] = "MaximizeObjective"


class MinimizeObjective(IdentityObjective):
    """Class returning the negative identity as reward.

    Attributes:
        w (float): float between zero and one for weighting the objective
        lower_bound (float, optional): Lower bound for normalizing the objective between zero and one. Defaults to zero.
        upper_bound (float, optional): Upper bound for normalizing the objective between zero and one. Defaults to one.
    """

    type: Literal["MinimizeObjective"] = "MinimizeObjective"

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
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

    type: Literal["DeltaObjective"] = "DeltaObjective"
    ref_point: float
    scale: float = 1

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """The call function returning a reward for passed x values

        Args:
            x (np.ndarray): An array of x values

        Returns:
            np.ndarray: The difference between reference and the x value as reward, might be scaled with a passed scaling value
        """
        return (self.ref_point - x) * self.scale


class SigmoidObjective(Objective, BotorchConstrainedObjective):
    """Base class for all sigmoid shaped objectives

    Attributes:
        w (float): float between zero and one for weighting the objective.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.
        tp (float): Turning point of the sigmoid function.
    """

    type: Literal["SigmoidObjective"] = "SigmoidObjective"
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


class AbstractTargetObjective(Objective):
    # TODO: add docstring to AbstractTargetObjective

    type: Literal["AbstractTargetObjective"] = "AbstractTargetObjective"
    w: TWeight = 1
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
    # TODO: add docstring to CloseToTargetObjective

    type: Literal["CloseToTargetObjective"] = "CloseToTargetObjective"
    exponent: float

    def __call__(self, x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        return (
            np.abs(x - self.target_value) ** self.exponent
            - self.tolerance**self.exponent
        )


class TargetObjective(AbstractTargetObjective, BotorchConstrainedObjective):
    """Class for objectives for optimizing towards a target value

    Attributes:
        w (float): float between zero and one for weighting the objective.
        target_value (float): target value that should be reached.
        tolerance (float): Tolerance for reaching the target. Has to be greater than zero.
        steepness (float): Steepness of the sigmoid function. Has to be greater than zero.

    """

    type: Literal["TargetObjective"] = "TargetObjective"
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


# TODO: check list of all objectives, possibly remove abstract classes
AnyObjective = Union[
    IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
    DeltaObjective,
    SigmoidObjective,
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    ConstantObjective,
    AbstractTargetObjective,
    CloseToTargetObjective,
    TargetObjective,
]

AnyAbstractObjective = Union[
    IdentityObjective,
    MaximizeObjective,
    MinimizeObjective,
    DeltaObjective,
    SigmoidObjective,
    MaximizeSigmoidObjective,
    MinimizeSigmoidObjective,
    ConstantObjective,
    AbstractTargetObjective,
    CloseToTargetObjective,
    TargetObjective,
    BotorchConstrainedObjective,
]
