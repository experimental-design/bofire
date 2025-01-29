from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from bofire.data_models.features.feature import Input, TTransform


class NumericalInput(Input):
    """Abstract base class for all numerical (ordinal) input features."""

    unit: Optional[str] = None

    @staticmethod
    def valid_transform_types() -> List:
        return []

    @property
    @abstractmethod
    def lower_bound(self) -> float:
        pass

    @property
    @abstractmethod
    def upper_bound(self) -> float:
        pass

    def to_unit_range(
        self,
        values: Union[pd.Series, np.ndarray],
        use_real_bounds: bool = False,
    ) -> Union[pd.Series, np.ndarray]:
        """Convert to the unit range between 0 and 1.

        Args:
            values (pd.Series): values to be transformed
            use_real_bounds (bool, optional): if True, use the bounds from the
                actual values else the bounds from the feature. Defaults to False.

        Raises:
            ValueError: If lower_bound == upper bound an error is raised

        Returns:
            pd.Series: transformed values.

        """
        if use_real_bounds:
            lower, upper = self.get_bounds(
                transform_type=None,
                values=values,  # type: ignore
            )
            lower = lower[0]
            upper = upper[0]
        else:
            lower, upper = self.lower_bound, self.upper_bound

        if lower == upper:
            raise ValueError("Fixed feature cannot be transformed to unit range.")

        allowed_range = upper - lower
        return (values - lower) / allowed_range

    def from_unit_range(
        self,
        values: Union[pd.Series, np.ndarray],
    ) -> Union[pd.Series, np.ndarray]:
        """Convert from unit range.

        Args:
            values (pd.Series): values to transform from.

        Raises:
            ValueError: if the feature is fixed raise a value error.

        Returns:
            pd.Series: _description_

        """
        if self.is_fixed():
            raise ValueError("Fixed feature cannot be transformed from unit range.")

        allowed_range = self.upper_bound - self.lower_bound

        return (values * allowed_range) + self.lower_bound

    def is_fixed(self):
        """Method to check if the feature is fixed

        Returns:
            Boolean: True when the feature is fixed, false otherwise.

        """
        return self.lower_bound == self.upper_bound

    def fixed_value(
        self,
        transform_type: Optional[TTransform] = None,
    ) -> Union[None, List[float]]:
        """Method to get the value to which the feature is fixed

        Returns:
            Float: Return the feature value or None if the feature is not fixed.

        """
        assert transform_type is None
        if self.is_fixed():
            return [self.lower_bound]
        return None

    def validate_experimental(self, values: pd.Series, strict=False) -> pd.Series:
        """Method to validate the experimental dataFrame

        Args:
            values (pd.Series): A dataFrame with experiments
            strict (bool, optional): Boolean to distinguish if the occurrence of fixed features in the dataset should be considered or not.
                Defaults to False.

        Raises:
            ValueError: when a value is not numerical
            ValueError: when there is no variation in a feature provided by the experimental data

        Returns:
            pd.Series: A dataFrame with experiments

        """
        try:
            values = pd.to_numeric(values, errors="raise").astype("float64")
        except ValueError:
            raise ValueError(
                f"not all values of input feature `{self.key}` are numerical",
            )

        values = values.astype("float64")
        if strict:
            lower, upper = self.get_bounds(transform_type=None, values=values)
            if lower == upper:
                raise ValueError(
                    f"No variation present or planned for feature {self.key}. Remove it.",
                )
        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Validate the suggested candidates for the feature.

        Args:
            values (pd.Series): suggested candidates for the feature

        Raises:
            ValueError: Error is raised when one of the values is not numerical.

        Returns:
            pd.Series: the original provided candidates

        """
        try:
            return pd.to_numeric(values, errors="raise").astype("float64")
        except ValueError:
            raise ValueError(
                f"not all values of input feature `{self.key}` are numerical",
            )
