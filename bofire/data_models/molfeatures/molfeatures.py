from abc import abstractmethod
from typing import Union, List, Optional

import numpy as np
import pandas as pd

from bofire.data_models.base import BaseModel
from bofire.data_models.features.feature import TDescriptors


class MolFeatures(BaseModel):
    """The base class for all molecular features"""

    type: str
    descriptors: TDescriptors = Optional[List[str]]

    @abstractmethod
    def __call__(self, values: pd.Series) -> pd.DataFrame:
        """Abstract method to define the call function for the class MolFeatures

        Args:
            values (pd.Series): An array of SMILES strings

        Returns:
            pd.DataFrame: Molecular features
        """
        pass
