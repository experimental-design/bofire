from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.objectives.objective import Objective
from bofire.data_models.types import Bounds


class IdentityObjective(Objective):
    """An objective returning the identity as reward.
    The return can be scaled, when a lower and upper bound are provided.
    """

    type: Literal["IdentityObjective"] = "IdentityObjective"
    bounds: Bounds = Field(
        default=[0, 1],
        description="Lower and upper bound used to normalize the objective onto the "
        "unit interval. The default of [0, 1] leaves the values unchanged.",
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
    """Rewards larger output values, without an upper limit.

    The reward is the output value itself, so improving from 9 to 10 counts exactly as
    much as improving from 1 to 2.

    Examples:
        >>> MaximizeObjective()
    """

    type: Literal["MaximizeObjective"] = "MaximizeObjective"

    def to_description(self) -> str:
        return "Maximize"


class MinimizeObjective(IdentityObjective):
    """Rewards smaller output values, without a lower limit.

    The reward is the negated output value, so improving from 2 to 1 counts exactly as
    much as improving from 10 to 9.

    Examples:
        >>> MinimizeObjective()
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
