from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from pydantic import Field
from typing_extensions import Annotated

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


# TODO: should this inherit from Objective?
class ConstrainedObjective:
    """This abstract class offers a convenience routine for transforming sigmoid based objectives to botorch output constraints."""


TGt0 = Annotated[float, Field(type=float, gt=0)]
TGe0 = Annotated[float, Field(type=float, ge=0)]
TWeight = Annotated[float, Field(type=float, gt=0, le=1)]
