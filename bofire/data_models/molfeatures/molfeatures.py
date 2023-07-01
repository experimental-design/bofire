from abc import abstractmethod
from typing import List

import pandas as pd
from pydantic import Field, validator

from bofire.data_models.base import BaseModel


class MolFeatures(BaseModel):
    """The base class for all molecular features"""

    type: str
    descriptors: List[str] = Field(default_factory=list)

    @abstractmethod
    def __call__(self, values: pd.Series) -> pd.DataFrame:
        """Abstract method to define the call function for the class MolFeatures

        Args:
            values (pd.Series): An array of SMILES strings

        Returns:
            pd.DataFrame: Molecular features
        """
        pass

    @validator("descriptors")
    def validate_descriptors(cls, descriptors):
        """validates that descriptors have unique names

        Args:
            categories (List[str]): List of descriptor names

        Raises:
            ValueError: when descriptors have non-unique names

        Returns:
            List[str]: List of the descriptors
        """
        if descriptors is not None:
            if len(descriptors) != len(set(descriptors)):
                raise ValueError("descriptors must be unique")
        return descriptors
