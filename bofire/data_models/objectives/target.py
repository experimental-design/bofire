from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from bofire.data_models.objectives.objective import (
    ConstrainedObjective,
    Objective,
    TGe0,
    TGt0,
    TWeight,
)


class CloseToTargetObjective(Objective):
    """Optimize towards a target value. It can be used as objective
    in multiobjective scenarios.

    Attributes:
        w (float): float between zero and one for weighting the objective.
        target_value (float): target value that should be reached.
        exponent (float): the exponent of the expression.

    """

    type: Literal["CloseToTargetObjective"] = "CloseToTargetObjective"
    w: TWeight = 1
    target_value: float
    exponent: float

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        return -1 * (np.abs(x - self.target_value) ** self.exponent)


class TargetObjective(Objective, ConstrainedObjective):
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
