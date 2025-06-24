import math
from typing import Annotated, ClassVar, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import Field, PositiveFloat, model_validator

from bofire.data_models.features.feature import Output, TTransform
from bofire.data_models.features.numerical import NumericalInput
from bofire.data_models.objectives.api import AnyObjective, MaximizeObjective
from bofire.data_models.types import Bounds


class ContinuousInput(NumericalInput):
    """Base class for all continuous input features.

    Attributes:
        bounds (Tuple[float, float]): A tuple that stores the lower and upper bound of the feature.
        stepsize (PositiveFloat, optional): Float indicating the allowed stepsize between lower and upper. Defaults to None.
        local_relative_bounds (Tuple[float, float], optional): A tuple that stores the lower and upper bounds relative to a reference value.
            Defaults to None.

    """

    type: Literal["ContinuousInput"] = "ContinuousInput"  # type: ignore
    order_id: ClassVar[int] = 1

    bounds: Bounds
    local_relative_bounds: Optional[
        Annotated[List[Annotated[float, Field(gt=0)]], Field(min_items=2, max_items=2)]  # type: ignore
    ] = None
    stepsize: Optional[PositiveFloat] = None

    @property
    def lower_bound(self) -> float:
        return self.bounds[0]

    @property
    def upper_bound(self) -> float:
        return self.bounds[1]

    @model_validator(mode="after")
    def validate_step_size(self):
        if self.stepsize is None:
            return self
        lower, upper = self.bounds
        if lower == upper and self.stepsize is not None:
            raise ValueError(
                "Stepsize cannot be provided for a fixed continuous input.",
            )
        range = upper - lower

        if range / self.stepsize < 1:
            raise ValueError(
                "Stepsize is too big for provided range.",
            )

        return self

    def _get_allowed_steps(self) -> List[float]:
        """Method to get the allowed steps of the feature.

        Returns:
            List[float]: The allowed steps of the feature

        """
        if self.stepsize is None:
            return []
        lower, upper = self.bounds

        n_steps = math.ceil((upper - lower) / self.stepsize)
        allowed_vals = np.linspace(lower, lower + n_steps * self.stepsize, n_steps + 1)
        # set the last value of allowed_vals to upper
        allowed_vals[-1] = upper
        return list(allowed_vals)

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

        steps = np.array(self._get_allowed_steps())

        return values.apply(lambda x: steps[np.argmin(np.abs(steps - x))])

    def is_fulfilled(self, values: pd.Series, noise: float = 10e-6) -> pd.Series:
        """Method to check if the values are within the bounds of the feature.

        Args:
            values: A series with values for the input feature.
            noise: A small value to allow for numerical errors. Defaults to 10e-6.

        Returns:
            A series with boolean values indicating if the input feature is fulfilled.
        """
        return (values >= self.lower_bound - noise) & (
            values <= self.upper_bound + noise
        )

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
                f"not all values of input feature `{self.key}`are larger than lower bound `{self.lower_bound}` ",
            )
        if (values > self.upper_bound + noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}`are smaller than upper bound `{self.upper_bound}` ",
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
            data=np.random.default_rng(seed=seed).uniform(
                self.lower_bound,
                self.upper_bound,
                n,
            ),
        )

    def get_bounds(  # type: ignore
        self,
        transform_type: Optional[TTransform] = None,
        values: Optional[pd.Series] = None,
        reference_value: Optional[float] = None,
    ) -> Tuple[List[float], List[float]]:
        assert transform_type is None
        if reference_value is not None and values is not None:
            raise ValueError("Only one can be used, `local_value` or `values`.")

        if values is None:
            if reference_value is None or self.is_fixed():
                return [self.lower_bound], [self.upper_bound]

            local_relative_bounds = self.local_relative_bounds or (
                math.inf,
                math.inf,
            )

            return [
                max(
                    reference_value - local_relative_bounds[0],
                    self.lower_bound,
                ),
            ], [
                min(
                    reference_value + local_relative_bounds[1],
                    self.upper_bound,
                ),
            ]

        lower = min(self.lower_bound, values.min())
        upper = max(self.upper_bound, values.max())
        return [lower], [upper]

    def __str__(self) -> str:
        """Method to return a string of lower and upper bound

        Returns:
            str: String of a list with lower and upper bound

        """
        return f"[{self.lower_bound},{self.upper_bound}]"


class ContinuousOutput(Output):
    """The base class for a continuous output feature

    Attributes:
        objective (objective, optional): objective of the feature indicating in which direction it should be optimized. Defaults to `MaximizeObjective`.

    """

    type: Literal["ContinuousOutput"] = "ContinuousOutput"  # type: ignore
    order_id: ClassVar[int] = 9
    unit: Optional[str] = None

    objective: Optional[AnyObjective] = Field(
        default_factory=lambda: MaximizeObjective(w=1.0),
    )

    def __call__(self, values: pd.Series, values_adapt: pd.Series) -> pd.Series:  # type: ignore
        if self.objective is None:
            return pd.Series(
                data=[np.nan for _ in range(len(values))],
                index=values.index,
                name=values.name,
            )
        return self.objective(values, values_adapt)  # type: ignore

    def validate_experimental(self, values: pd.Series) -> pd.Series:
        try:
            values = pd.to_numeric(values, errors="raise").astype("float64")
        except ValueError:
            raise ValueError(
                f"not all values of input feature `{self.key}` are numerical",
            )
        return values

    def __str__(self) -> str:
        return "ContinuousOutputFeature"
