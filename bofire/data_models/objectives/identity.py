from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.objectives.objective import Objective, TWeight
from bofire.data_models.types import Bounds


class IdentityObjective(Objective):
    """An objective returning the identity as reward.
    The return can be scaled, when a lower and upper bound are provided.
    """

    type: Literal["IdentityObjective"] = "IdentityObjective"
    w: TWeight = Field(
        default=1,
        description="Relative weight of this objective, between zero and one.",
    )
    bounds: Bounds = Field(
        default=[0, 1],
        description="Lower and upper bound used to normalize the objective between "
        "zero and one.",
    )

    @property
    def lower_bound(self) -> float:
        return self.bounds[0]

    @property
    def upper_bound(self) -> float:
        return self.bounds[1]

    @field_validator("bounds")
    @classmethod
    def validate_lower_upper(cls, bounds):
        """Validation function to ensure that lower bound is strictly below upper bound.


        Args:
            values (Dict): The attributes of the class

        Raises:
            ValueError: when bounds are not strictly increasing.

        Returns:
            Dict: The attributes of the class

        """
        if bounds[0] >= bounds[1]:
            raise ValueError(
                f"lower bound must be < upper bound, got {bounds[0]} >= {bounds[1]}",
            )
        return bounds

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a reward for passed x values

        Args:
            x (np.ndarray): An array of x values
            x_adapt (Optional[np.ndarray], optional): An array of x values which are used to
                update the objective parameters on the fly. Defaults to None.

        Returns:
            np.ndarray: The identity as reward, might be normalized to the passed lower and upper bounds

        """
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)


class MaximizeObjective(IdentityObjective):
    """Child class from the identity function without modifications, since the parent class is already defined as maximization

    The default objective for an output that should be as large as possible. Use
    `MinimizeObjective` for the opposite, or `CloseToTargetObjective` to aim at a
    specific value.

    Examples:
        >>> MaximizeObjective(bounds=[0.0, 100.0])
    """

    type: Literal["MaximizeObjective"] = "MaximizeObjective"

    def to_description(self) -> str:
        return "Maximize"


class MinimizeObjective(IdentityObjective):
    """Class returning the negative identity as reward.

    For an output that should be as small as possible. Use `MaximizeObjective` for the
    opposite, or `CloseToTargetObjective` to aim at a specific value.

    Examples:
        >>> MinimizeObjective(bounds=[0.0, 100.0])
    """

    type: Literal["MinimizeObjective"] = "MinimizeObjective"

    def to_description(self) -> str:
        return "Minimize"

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a reward for passed x values

        Args:
            x (np.ndarray): An array of x values
            x_adapt (Optional[np.ndarray], optional): An array of x values which are used to
                update the objective parameters on the fly. Defaults to None.

        Returns:
            np.ndarray: The negative identity as reward, might be normalized to the passed lower and upper bounds

        """
        return -1.0 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound)
