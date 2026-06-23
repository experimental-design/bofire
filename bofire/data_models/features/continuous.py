import math
from typing import Annotated, ClassVar, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import Field, PositiveFloat, model_validator
from pydantic.fields import FieldInfo

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
        allow_zero (bool): A boolean indicating if the input feature can take inactive values.
            Useful for features that take values between `bounds`, but can also take a value of 0.
            One may choose to use a conditional kernel for this, if taking a value of 0
            represents a distinct behaviour from non-zero values.

    """

    type: Literal["ContinuousInput"] = "ContinuousInput"
    order_id: ClassVar[int] = 1

    bounds: Bounds
    local_relative_bounds: Optional[
        Annotated[List[PositiveFloat], Field(min_length=2, max_length=2)]
    ] = None
    stepsize: Optional[PositiveFloat] = None
    allow_zero: bool = False

    @property
    def lower_bound(self) -> float:
        return self.bounds[0]

    @property
    def upper_bound(self) -> float:
        return self.bounds[1]

    @property
    def is_semicontinuous(self) -> bool:
        """True iff the feasible region is the disconnected union
        ``{0} ∪ [lb, ub]`` — i.e., ``allow_zero=True`` *and* a strictly
        positive lower bound. Used by NChooseK pruning and the BoTorch
        optimizer-routing logic to detect features that need the
        semi-continuous handling path.
        """
        return self.allow_zero and self.bounds[0] > 0

    def _description_prefix(self) -> str:
        """Leading description string identifying this feature kind."""
        return f"Continuous, bounds [{self.bounds[0]}, {self.bounds[1]}]"

    def _extra_description_parts(self) -> List[str]:
        """Optional extras appended after the prefix, before context."""
        return []

    def to_pydantic_field(self) -> Tuple[type, FieldInfo]:
        """Return ``(float, Field(ge=..., le=..., description=...))```.

        Subclasses customize the output by overriding ``_description_prefix``
        and/or ``_extra_description_parts``.

        Example::

            >>> feat = ContinuousInput(key="temp", bounds=(20.0, 200.0), context="Temperature in C")
            >>> field_type, field_info = feat.to_pydantic_field()
            >>> # field_type = float
            >>> # field_info has ge=20.0, le=200.0
            >>> # description = "Continuous, bounds [20.0, 200.0] — Temperature in C"
        """
        desc_parts = [self._description_prefix(), *self._extra_description_parts()]
        lower = self.bounds[0]
        if self.allow_zero:
            lower = min(0.0, lower)
            desc_parts.append("can also be 0 (inactive)")
        if self.context:
            desc_parts.append(self.context)
        return (
            float,
            Field(ge=lower, le=self.bounds[1], description=" — ".join(desc_parts)),
        )

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

    @model_validator(mode="after")
    def validate_allow_zero(self):
        if not self.allow_zero:
            return self
        lower, upper = self.bounds
        # When both bounds are exactly zero the feature is pinned to zero
        # (e.g. by NChooseK deactivation), which is always valid regardless
        # of allow_zero.
        if lower == 0.0 and upper == 0.0:
            return self
        if lower <= 0.0 <= upper:
            raise ValueError(
                "If `allow_zero==True`, then zero must not lie within the bounds."
            )
        # A positively-fixed feature with allow_zero=True would have the
        # disjoint feasible set ``{0} ∪ {v}`` — a 2-point discrete set
        # masquerading as a semi-continuous fixed feature. The intent is
        # ambiguous (is the feature really fixed, or really binary?), so
        # we reject it. Users who want a 2-point set should use
        # ``DiscreteInput(values=[0, v])``.
        if lower == upper:
            raise ValueError(
                "`allow_zero=True` is not compatible with a positively-fixed "
                "feature (`bounds[0] == bounds[1] > 0`). The resulting feasible "
                "set `{0, v}` is a 2-point discrete set, not a continuous "
                "feature. Use `DiscreteInput(values=[0, v])` instead."
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

        return values.apply(
            lambda x: steps[np.argmin(np.abs(steps - x))]
        )  # ty: ignore[invalid-return-type]

    def is_fulfilled(self, values: pd.Series, noise: float = 10e-6) -> pd.Series:
        """Method to check if the values are within the bounds of the feature.

        Args:
            values: A series with values for the input feature.
            noise: A small value to allow for numerical errors. Defaults to 10e-6.

        Returns:
            A series with boolean values indicating if the input feature is fulfilled.
        """
        within_bounds = (values >= self.lower_bound - noise) & (
            values <= self.upper_bound + noise
        )
        zero_and_allowed = (values.abs() <= noise) & self.allow_zero
        return within_bounds | zero_and_allowed

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
        non_zero_idcs = (values.abs() > noise) | (not self.allow_zero)
        if (values[non_zero_idcs] < self.lower_bound - noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}` are larger than lower bound `{self.lower_bound}` ",
            )
        if (values[non_zero_idcs] > self.upper_bound + noise).any():
            raise ValueError(
                f"not all values of input feature `{self.key}` are smaller than upper bound `{self.upper_bound}` ",
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

    def get_bounds(
        self,
        transform_type: Optional[TTransform] = None,
        values: Optional[pd.Series] = None,
        reference_value: Optional[float] = None,
        relax_allow_zero: bool = False,
    ) -> Tuple[List[float], List[float]]:
        assert transform_type is None
        if reference_value is not None and values is not None:
            raise ValueError("Only one can be used, `local_value` or `values`.")

        # Effective lower bound: 0 for semi-continuous features when the
        # caller asks for the convex-relaxation view. (Fixed semi-
        # continuous features are forbidden by the `allow_zero` validator,
        # so the `is_semicontinuous` check below implies `not is_fixed()`.)
        effective_lower = self.lower_bound
        if relax_allow_zero and self.is_semicontinuous:
            effective_lower = 0.0

        if values is None:
            if reference_value is None or self.is_fixed():
                return [effective_lower], [self.upper_bound]

            local_relative_bounds = self.local_relative_bounds or (
                math.inf,
                math.inf,
            )

            return [
                max(
                    reference_value - local_relative_bounds[0],
                    effective_lower,
                ),
            ], [
                min(
                    reference_value + local_relative_bounds[1],
                    self.upper_bound,
                ),
            ]

        lower = min(effective_lower, values.min())
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

    type: Literal["ContinuousOutput"] = "ContinuousOutput"
    order_id: ClassVar[int] = 9
    unit: Optional[str] = None

    objective: Optional[AnyObjective] = Field(
        default_factory=lambda: MaximizeObjective(w=1.0),
    )

    def to_description(self) -> str:
        """Return a human-readable description combining objective and context.

        Example::

            >>> feat = ContinuousOutput(key="yield", objective=MaximizeObjective(w=1.0), context="Target >90%")
            >>> feat.to_description()
            'yield: Maximize — Target >90%'
        """
        parts = [self.key]
        if self.objective is not None:
            parts.append(self.objective.to_description())
        if self.context:
            parts.append(self.context)
        return ": ".join(parts[:2]) + (" — " + parts[2] if len(parts) > 2 else "")

    def __call__(self, values: pd.Series, values_adapt: pd.Series) -> pd.Series:
        if self.objective is None:
            return pd.Series(
                data=[np.nan for _ in range(len(values))],
                index=values.index,
                name=values.name,
            )
        return self.objective(values, values_adapt)  # ty: ignore[invalid-return-type]

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
