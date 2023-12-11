from typing import ClassVar, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import Field, root_validator, validator

from bofire.data_models.features.feature import Output
from bofire.data_models.features.numerical import NumericalInput
from bofire.data_models.objectives.api import AnyObjective, MaximizeObjective


class ContinuousInput(NumericalInput):
    """Base class for all continuous input features.

    Attributes:
        bounds (Tuple[float, float]): A tuple that stores the lower and upper bound of the feature.
        stepsize (float, optional): Float indicating the allowed stepsize between lower and upper. Defaults to None.
    """

    type: Literal["ContinuousInput"] = "ContinuousInput"
    order_id: ClassVar[int] = 1

    bounds: Tuple[float, float]
    stepsize: Optional[float] = None

    @property
    def lower_bound(self) -> float:
        return self.bounds[0]

    @property
    def upper_bound(self) -> float:
        return self.bounds[1]

    @validator("stepsize")
    def validate_step_size(cls, v, values):
        if v is None:
            return v
        lower, upper = values["bounds"]
        if lower == upper and v is not None:
            raise ValueError(
                "Stepsize cannot be provided for a fixed continuous input."
            )
        range = upper - lower
        if np.arange(lower, upper + v, v)[-1] != upper:
            raise ValueError(
                f"Stepsize of {v} does not match the provided interval [{lower},{upper}]."
            )
        if range // v == 1:
            raise ValueError("Stepsize is too big, only one value allowed.")
        return v

    def round(self, values: pd.Series) -> pd.Series:
        """Round values to the stepsize of the feature. If no stepsize is provided return the
        provided values.

        Args:
            values (pd.Series): The values that should be rounded.

        Returns:
            pd.Series: The rounded values
        """
        if self.stepsize is None:
            return values
        self.validate_candidental(values=values)
        allowed_values = np.arange(
            self.lower_bound, self.upper_bound + self.stepsize, self.stepsize
        )
        idx = abs(values.values.reshape([len(values), 1]) - allowed_values).argmin(
            axis=1
        )
        return pd.Series(
            data=self.lower_bound + idx * self.stepsize, index=values.index
        )

    @root_validator(pre=False, skip_on_failure=True)
    def validate_lower_upper(cls, values):
        """Validates that the lower bound is lower than the upper bound

        Args:
            values (Dict): Dictionary with attributes key, lower and upper bound

        Raises:
            ValueError: when the lower bound is higher than the upper bound

        Returns:
            Dict: The attributes as dictionary
        """
        if values["bounds"][0] > values["bounds"][1]:
            raise ValueError(
                f'lower bound must be <= upper bound, got {values["lower_bound"]} > {values["upper_bound"]}'
            )
        return values

    def validate_candidental(self, values: pd.Series) -> pd.Series:
        """Method to validate the suggested candidates

        Args:
            values (pd.Series): A dataFrame with candidates

        Raises:
            ValueError: when non numerical values are passed
            ValueError: when values are larger than the upper bound of the feature
            ValueError: when values are lower than the lower bound of the feature

        Returns:
            pd.Series: The passed dataFrame with candidates
        """

        noise = 10e-6
        values = super().validate_candidental(values)
        if (values < self.lower_bound - noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}`are larger than lower bound `{self.lower_bound}` "
            )
        if (values > self.upper_bound + noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}`are smaller than upper bound `{self.upper_bound}` "
            )
        return values

    def sample(self, n: int) -> pd.Series:
        """Draw random samples from the feature.

        Args:
            n (int): number of samples.

        Returns:
            pd.Series: drawn samples.
        """
        return pd.Series(
            name=self.key,
            data=np.random.uniform(self.lower_bound, self.upper_bound, n),
        )

    def __str__(self) -> str:
        """Method to return a string of lower and upper bound

        Returns:
            str: String of a list with lower and upper bound
        """
        return f"[{self.lower_bound},{self.upper_bound}]"


class ContinuousOutput(Output):
    """The base class for a continuous output feature

    Attributes:
        objective (objective, optional): objective of the feature indicating in which direction it should be optimzed. Defaults to `MaximizeObjective`.
    """

    type: Literal["ContinuousOutput"] = "ContinuousOutput"
    order_id: ClassVar[int] = 7
    unit: Optional[str] = None

    objective: Optional[AnyObjective] = Field(
        default_factory=lambda: MaximizeObjective(w=1.0)
    )

    def __call__(self, values: pd.Series) -> pd.Series:
        if self.objective is None:
            return pd.Series(
                data=[np.nan for _ in range(len(values))],
                index=values.index,
                name=values.name,
            )
        return self.objective(values)  # type: ignore

    def validate_experimental(self, values: pd.Series) -> pd.Series:
        try:
            values = pd.to_numeric(values, errors="raise").astype("float64")
        except ValueError:
            raise ValueError(
                f"not all values of input feature `{self.key}` are numerical"
            )
        return values

    def __str__(self) -> str:
        return "ContinuousOutputFeature"
