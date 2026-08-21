from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.objectives.objective import (
    ConstrainedObjective,
    Objective,
    TGt0,
)


class SigmoidObjective(Objective, ConstrainedObjective):
    """Base class for all sigmoid shaped objectives

    A sigmoid objective expresses a threshold rather than a direction: the reward runs
    from 0 to 1 across a transition centred on `tp`, and is flat on either side. Values
    far beyond the threshold therefore earn no additional credit.
    """

    steepness: TGt0 = Field(
        description="Steepness of the sigmoid. Larger values make the transition "
        "between unrewarded and rewarded values sharper, approaching a hard step.",
    )
    tp: float = Field(
        description="Turning point of the sigmoid, i.e. the threshold at which the "
        "reward is one half.",
    )


class MaximizeSigmoidObjective(SigmoidObjective):
    """Maximize an output up to a threshold, after which it stops counting.

    The reward rises from 0 well below `tp` to 1 well above it. Because it saturates,
    strategies commonly use this as an output constraint -- "this response must exceed
    `tp`" -- rather than as the quantity being optimized.

    Examples:
        >>> MaximizeSigmoidObjective(tp=5.0, steepness=10.0)
    """

    type: Literal["MaximizeSigmoidObjective"] = "MaximizeSigmoidObjective"

    def to_description(self) -> str:
        raise NotImplementedError

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values
            x_adapt (np.ndarray): An array of x values which are used to update the objective parameters on the fly.

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.

        """
        return 1 / (1 + np.exp(-1 * self.steepness * (x - self.tp)))


class MovingMaximizeSigmoidObjective(SigmoidObjective):
    """Class for a maximizing sigmoid objective with a moving turning point that depends on so far observed x values.

    Note that `tp` is interpreted differently here than in the other sigmoid
    objectives: it is a *relative* turning point, and the effective one is obtained by
    adding the maximum of the observed x values to it. Use this when the target is
    "improve on the best result so far" rather than a fixed threshold.

    Examples:
        >>> MovingMaximizeSigmoidObjective(tp=0.0, steepness=10.0)
    """

    type: Literal["MovingMaximizeSigmoidObjective"] = "MovingMaximizeSigmoidObjective"

    def to_description(self) -> str:
        raise NotImplementedError

    def get_adjusted_tp(self, x: Union[pd.Series, np.ndarray]) -> float:
        """Get the adjusted turning point for the sigmoid function.

        Args:
            x (np.ndarray): An array of x values

        Returns:
            float: The adjusted turning point for the sigmoid function.

        """
        return x.max() + self.tp

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Union[pd.Series, np.ndarray],
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values
            x_adapt (np.ndarray): An array of x values which are used to update the objective parameters on the fly.

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.

        """
        return 1 / (
            1 + np.exp(-1 * self.steepness * (x - self.get_adjusted_tp(x_adapt)))
        )


class MinimizeSigmoidObjective(SigmoidObjective):
    """Minimize an output down to a threshold, after which it stops counting.

    The reward rises from 0 well above `tp` to 1 well below it. Because it saturates,
    strategies commonly use this as an output constraint -- "this response must stay
    below `tp`" -- rather than as the quantity being optimized.

    Examples:
        >>> MinimizeSigmoidObjective(tp=5.0, steepness=10.0)
    """

    type: Literal["MinimizeSigmoidObjective"] = "MinimizeSigmoidObjective"

    def to_description(self) -> str:
        raise NotImplementedError

    def __call__(
        self,
        x: Union[pd.Series, np.ndarray],
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """The call function returning a sigmoid shaped reward for passed x values.

        Args:
            x (np.ndarray): An array of x values
            x_adapt (np.ndarray): An array of x values which are used to update the objective parameters on the fly.

        Returns:
            np.ndarray: A reward calculated with a sigmoid function. The stepness and the tipping point can be modified via passed arguments.

        """
        return 1 - 1 / (1 + np.exp(-1 * self.steepness * (x - self.tp)))
