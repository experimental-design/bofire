import math
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from pydantic import root_validator

from bofire.data_models.features.continuous import ContinuousInput


class RelaxableBinaryInput(ContinuousInput):
    """Base class for all binary input features. It behaves like a continuous inputs to allow for easy relaxation.
    It is not a true binary variable as it does not allow the variable to be either 0 or 1 while prohibiting all values
    in between.

    """

    def __init__(self, **kwargs):
        if "bounds" in kwargs and kwargs["bounds"] != (0, 1):
            warnings.warn("bounds must be set to (0, 1). Readjusting the bounds...")
            kwargs["bounds"] = (0, 1)
            super().__init__(**kwargs)
        else:
            bounds = (0, 1)
            super().__init__(bounds=bounds, **kwargs)

    @root_validator(pre=False, skip_on_failure=True)
    def validate_lower_upper(cls, values):
        """Validates that lower bound is set to 0 and upper bound set to 1

        Args:
            values (Dict): Dictionary with attributes key, lower and upper bound

        Raises:
            ValueError: when the lower bound is higher than the upper bound

        Returns:
            Dict: The attributes as dictionary
        """

        if values["bounds"][0] != values["bounds"][1] and (
            values["bounds"][0] != 0 or values["bounds"][1] != 1
        ):
            raise ValueError(
                f'if variable is relaxed, lower bound must be 0 and upper bound 1, got {values["bounds"][0]}, {values["bounds"][1]}'
            )
        elif values["bounds"][0] == values["bounds"][1] and not (
            values["bounds"][0] == 0 or values["bounds"][0] == 1
        ):
            raise ValueError(
                f"if variable is fixed, lower and upper bound must be equal and both either 0 or 1,"
                f' got {values["bounds"][0]}, {values["bounds"][1]}'
            )

        return values


class RelaxableDiscreteInput(ContinuousInput):
    """Feature with discrete ordinal values allowed in the optimization.

    Attributes:
        key(str): key of the feature.
        values(List[float]): the allowed discrete values during the optimization.
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

    def equal_range_split(
        self, lower_bound: float, upper_bound: float
    ) -> Tuple[float, float]:
        """
        Determines the two identical elements x such that the intervals (lower_bound, x) and (x, upper_bound)
        are of equal length
        Args:
            lower_bound: inclusive lower bound
            upper_bound: inclusive upper bound

        Returns: tuple of floats which split the interval in half

        """
        x = (upper_bound - lower_bound) / 2 + lower_bound
        return x, x

    def equal_count_split(
        self, lower_bound: float, upper_bound: float
    ) -> Tuple[float, float]:
        """
        Determines the two elements x and y such that the intervals (lower_bound, x) and (y, upper_bound)
        have the same number of elements regarding the values of the discrete variable
        Args:
            lower_bound: inclusive lower bound
            upper_bound: inclusive upper bound

        Returns: tuple of floats which split the interval in half

        """
        self.values.sort()
        sub_list = [elem for elem in self.values if lower_bound <= elem <= upper_bound]

        size = len(sub_list)
        if size % 2 == 0:
            lower_index = size / 2 - 1
            upper_index = size / 2
        elif size == 1:
            return sub_list[0], sub_list[0]
        else:
            lower_index = math.floor(size / 2)
            upper_index = math.ceil(size / 2)

        lower_index = int(lower_index)
        upper_index = int(upper_index)

        return sub_list[lower_index], sub_list[upper_index]
