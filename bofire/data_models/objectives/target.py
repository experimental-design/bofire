from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.objectives.objective import (
    ConstrainedObjective,
    Objective,
    TGe0,
    TGt0,
)


class CloseToTargetObjective(Objective):
    """Optimize towards a target value. It can be used as objective
    in multiobjective scenarios.

    Unlike `TargetObjective`, this is an unconstrained objective: the reward decays
    smoothly away from the target rather than defining a feasible window.

    Examples:
        >>> CloseToTargetObjective(target_value=7.0, exponent=2.0)
    """

    type: Literal["CloseToTargetObjective"] = "CloseToTargetObjective"
    target_value: float = Field(description="Target value that should be reached.")
    exponent: float = Field(
        description="Exponent applied to the distance from the target. Larger values "
        "penalize deviations more sharply.",
    )

    def to_description(self) -> str:
        raise NotImplementedError

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        return -1 * (np.abs(x - self.target_value) ** self.exponent)


class TargetObjective(Objective, ConstrainedObjective):
    """Class for objectives for optimizing towards a target value

    A constrained objective: the reward is the product of two sigmoids, forming a
    plateau of width `tolerance` around the target. Use `CloseToTargetObjective` when
    no feasible window is intended.

    Examples:
        >>> TargetObjective(target_value=7.0, tolerance=0.5, steepness=10.0)
    """

    type: Literal["TargetObjective"] = "TargetObjective"
    target_value: float = Field(description="Target value that should be reached.")
    tolerance: TGe0 = Field(
        description="Half-width of the accepted window around the target value.",
    )
    steepness: TGt0 = Field(
        description="Steepness of the sigmoids bounding the window. Larger values "
        "make the transition to infeasible sharper.",
    )

    def to_description(self) -> str:
        raise NotImplementedError

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a reward for passed x values.

        Args:
            x (np.array): An array of x values
            x_adapt (Optional[np.ndarray], optional): An array of x values which are used to
                update the objective parameters on the fly. Defaults to None.

        Returns:
            np.array: An array of reward values calculated by the product of two sigmoidal shaped functions resulting in a maximum at the target value.

        """
        return (
            1
            / (
                1
                + np.exp(
                    -1 * self.steepness * (x - (self.target_value - self.tolerance)),
                )
            )
            * (
                1
                - 1
                / (
                    1.0
                    + np.exp(
                        -1
                        * self.steepness
                        * (x - (self.target_value + self.tolerance)),
                    )
                )
            )
        )
