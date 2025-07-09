from typing import ClassVar, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import field_validator

from bofire.data_models.features.feature import TTransform
from bofire.data_models.features.numerical import NumericalInput
from bofire.data_models.types import DiscreteVals


class DiscreteInput(NumericalInput):
    """Feature with discretized ordinal values allowed in the optimization.

    Attributes:
        key(str): key of the feature.
        values(List[float]): the discretized allowed values during the optimization.

    """

    type: Literal["DiscreteInput"] = "DiscreteInput"
    order_id: ClassVar[int] = 3

    values: DiscreteVals
    rtol: float = 1e-7

    @field_validator("values")
    @classmethod
    def validate_values_unique(cls, values):
        """Validates that provided values are unique.

        Args:
            values (List[float]): List of values

        Raises:
            ValueError: when values are non-unique.
            ValueError: when values contains only one entry.
            ValueError: when values is empty.

        Returns:
            List[values]: Sorted list of values

        """
        if len(values) != len(set(values)):
            raise ValueError("Discrete values must be unique")
        if len(values) == 1:
            raise ValueError(
                "Fixed discrete inputs are not supported. Please use a fixed continuous input.",
            )
        if len(values) == 0:
            raise ValueError("No values defined.")
        return sorted(values)

    @property
    def lower_bound(self) -> float:
        """Lower bound of the set of allowed values"""
        return min(self.values)

    @property
    def upper_bound(self) -> float:
        """Upper bound of the set of allowed values"""
        return max(self.values)

    def is_fulfilled(self, values: pd.Series) -> pd.Series:
        """Method to check if the values are close to the discrete values.

        Args:
            values: A series with values for the input feature.

        Returns:
            A series with boolean values indicating if the input feature is fulfilled.
        """
        return pd.Series(
            np.array(
                [
                    np.array(
                        [np.isclose(x, y, rtol=self.rtol) for x in self.values]
                    ).any()
                    for y in values.to_numpy()
                ]
            ),
            index=values.index,
        )

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the provided candidates.

        Args:
            values (pd.Series): suggested candidates for the feature

        Raises:
            ValueError: Raises error when one of the provided values is not contained in the list of allowed values.

        Returns:
            pd.Series: suggested candidates for the feature

        """
        values = super().validate_candidental(values)
        candidates_close_to_allowed_values = (
            np.array(
                [
                    np.array(
                        [np.isclose(x, y, rtol=self.rtol) for x in self.values]
                    ).any()
                    for y in values.to_numpy()
                ]
            )
        ).all()
        if not candidates_close_to_allowed_values:
            raise ValueError(
                f"Not allowed values in candidates for feature {self.key}.",
            )
        return values

    def sample(self, n: int, seed: Optional[int] = None) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.
            seed (int, optional): random seed. Defaults to None.

        Returns:
            pd.Series: drawn samples.

        """
        return pd.Series(
            name=self.key,
            data=np.random.default_rng(seed=seed).choice(self.values, n),
        )

    def from_continuous(self, values: pd.DataFrame) -> pd.Series:
        """Rounds continuous values to the closest discrete ones.

        Args:
            values (pd.DataFrame): Dataframe with continuous entries.

        Returns:
            pd.Series: Series with discrete values.

        """
        s = pd.DataFrame(
            data=np.abs(
                values[self.key].to_numpy()[:, np.newaxis] - np.array(self.values),
            ),
            columns=self.values,
            index=values.index,
        ).idxmin(1)
        s.name = self.key
        return s

    def get_bounds(
        self,
        transform_type: Optional[TTransform] = None,
        values: Optional[pd.Series] = None,
        reference_value: Optional[float] = None,
    ) -> Tuple[List[float], List[float]]:
        assert transform_type is None
        if values is None:
            return [self.lower_bound], [self.upper_bound]
        lower = min(self.lower_bound, values.min())
        upper = max(self.upper_bound, values.max())
        return [lower], [upper]
