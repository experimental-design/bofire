from abc import abstractmethod
from typing import Annotated, Union

import numpy as np
import pandas as pd
from pydantic import Field

from bofire.data_models.base import BaseModel


class MolFeatures(BaseModel):
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
