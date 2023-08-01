from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import root_validator

from bofire.data_models.objectives.objective import Objective, TWeight


class IdentityObjective(Objective):
    """An objective returning the identity as reward.
    The return can be scaled, when a lower and upper bound are provided.

    Attributes:
        w (float): float between zero and one for weighting the objective
        bounds (Tuple[float], optional): Bound for normalizing the objective between zero and one. Defaults to (0,1).
    """

    type: Literal["IdentityObjective"] = "IdentityObjective"
    w: TWeight = 1
    bounds: Tuple[float, float] = (0, 1)

    @property
    def lower_bound(self) -> float:
        return self.bounds[0]

    @property
    def upper_bound(self) -> float:
        return self.bounds[1]

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
        if values["bounds"][0] > values["bounds"][1]:
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
        bounds (Tuple[float], optional): Bound for normalizing the objective between zero and one. Defaults to (0,1).
    """

    type: Literal["MaximizeObjective"] = "MaximizeObjective"


class MinimizeObjective(IdentityObjective):
    """Class returning the negative identity as reward.

    Attributes:
        w (float): float between zero and one for weighting the objective
        bounds (Tuple[float], optional): Bound for normalizing the objective between zero and one. Defaults to (0,1).
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
