from abc import abstractmethod
from typing import Annotated, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field
from torch import Tensor

from bofire.data_models.base import BaseModel


class Objective(BaseModel):
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


# TODO: should this inherit from Objective?
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


TGt0 = Annotated[float, Field(type=float, gt=0)]
TGe0 = Annotated[float, Field(type=float, ge=0)]
TWeight = Annotated[float, Field(type=float, gt=0, le=1)]
