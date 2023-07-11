from typing import List

import numpy as np
import pandas as pd

from bofire.data_models.features.continuous import ContinuousInput


class ContinuousDiscreteInput(ContinuousInput):
    """Feature with discretized ordinal values allowed in the optimization.

    Attributes:
        key(str): key of the feature.
        values(List[float]): the discretized allowed values during the optimization.
    """

    values: List[float]

    def __init__(self, **kwargs):
        super().__init__(bounds=(0, 1), **kwargs)
        self.bounds = (self.lower_bound, self.upper_bound)

    @property
    def lower_bound(self) -> float:
        """Lower bound of the set of allowed values"""
        return min(self.values)

    @property
    def upper_bound(self) -> float:
        """Upper bound of the set of allowed values"""
        return max(self.values)

    @lower_bound.setter
    def lower_bound(self, lb: float):
        self.values = [val for val in self.values if val >= lb]

    @upper_bound.setter
    def upper_bound(self, ub: float):
        self.values = [val for val in self.values if val <= ub]

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(name=self.key, data=np.random.choice(self.values, n))
