from abc import abstractmethod
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import pydantic
from pydantic import Field

from bofire.data_models.objectives.identity import IdentityObjective


class DesirabilityObjective(IdentityObjective):
    """Abstract class for desirability objectives. Works as Identity Objective"""

    type: Literal["DesirabilityObjective"] = "DesirabilityObjective"
    clip: bool = Field(
        default=True,
        description="Whether to clip the desirability outside `bounds`. If disabled, "
        "the desirability continues past the bounds and all log shape factors must be "
        "zero.",
    )

    @pydantic.model_validator(mode="after")
    def validate_clip(self):
        if self.clip:
            return self

        log_shapes = {
            key: val
            for (key, val) in self.__dict__.items()
            if key.startswith("log_shape_factor")
        }
        for key, log_shape_ in log_shapes.items():
            if log_shape_ != 0:
                raise ValueError(
                    f"Log shape factor {key} must be zero if clip is False."
                )
        return self

    def __call__(
        self, x: Union[pd.Series, np.ndarray], x_adapt
    ) -> Union[pd.Series, np.ndarray]:
        """Wrapper function for to call numpy and torch functions with series
        or numpy arrays. matches __call__ signature of objectives."""
        if isinstance(x, pd.Series):
            s: pd.Series = x
            return pd.Series(self.call_numpy(s.to_numpy()), name=s.name)

        return self.call_numpy(x)

    @abstractmethod
    def call_numpy(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class IncreasingDesirabilityObjective(DesirabilityObjective):
    """An objective returning a reward the scaled identity, but trimmed at the bounds:

        d = ((x - lower_bound) / (upper_bound - lower_bound))^t

    if clip is True, the reward is zero for x < lower_bound and one for x > upper_bound.

    where:

        t = exp(log_shape_factor)

    Note, that with clipping the reward is always between zero and one.

    Below `bounds[0]` the desirability is 0 (or negative if `clip` is disabled), above
    `bounds[1]` it is 1 (or greater). Use `DecreasingDesirabilityObjective` for the
    opposite direction.

    Examples:
        >>> IncreasingDesirabilityObjective(bounds=[0.0, 10.0])
    """

    type: Literal["IncreasingDesirabilityObjective"] = "IncreasingDesirabilityObjective"
    log_shape_factor: float = Field(
        default=0.0,
        description="Logarithm of the shape factor: whether the interpolation between "
        "the lower and the upper bound is linear (=0), convex (>0) or concave (<0).",
    )

    def to_description(self) -> str:
        raise NotImplementedError

    def call_numpy(
        self,
        x: np.ndarray,
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        y = np.zeros(x.shape)
        if self.clip:
            y[x < self.lower_bound] = 0.0
            y[x > self.upper_bound] = 1.0
            between = (x >= self.lower_bound) & (x <= self.upper_bound)
        else:
            between = np.full(x.shape, True)

        t = np.exp(self.log_shape_factor)

        y[between] = np.power(
            (x[between] - self.lower_bound) / (self.upper_bound - self.lower_bound), t
        )

        return y


class DecreasingDesirabilityObjective(DesirabilityObjective):
    """An objective returning a reward the negative, shifted scaled identity, but trimmed at the bounds:

        d = ((upper_bound - x) / (upper_bound - lower_bound))^t

    where:

        t = exp(log_shape_factor)

    Note, that with clipping the reward is always between zero and one.

    Below `bounds[0]` the desirability is 1 (or greater if `clip` is disabled), above
    `bounds[1]` it is 0 (or negative). Use `IncreasingDesirabilityObjective` for the
    opposite direction.

    Examples:
        >>> DecreasingDesirabilityObjective(bounds=[0.0, 10.0])
    """

    type: Literal["DecreasingDesirabilityObjective"] = "DecreasingDesirabilityObjective"
    log_shape_factor: float = Field(
        default=0.0,
        description="Logarithm of the shape factor: whether the interpolation between "
        "the lower and the upper bound is linear (=0), convex (>0) or concave (<0).",
    )

    def to_description(self) -> str:
        raise NotImplementedError

    def call_numpy(
        self,
        x: np.ndarray,
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        y = np.zeros(x.shape)
        if self.clip:
            y[x < self.lower_bound] = 1.0
            y[x > self.upper_bound] = 0.0
            between = (x >= self.lower_bound) & (x <= self.upper_bound)
        else:
            between = np.full(x.shape, True)

        t = np.exp(self.log_shape_factor)

        y[between] = np.power(
            (self.upper_bound - x[between]) / (self.upper_bound - self.lower_bound), t
        )

        return y


class PeakDesirabilityObjective(DesirabilityObjective):
    """
    A piecewise (linear or convex/concave) objective that increases from the lower bound
    to the peak position and decreases from the peak position to the upper bound.

    The desirability is 0 outside `bounds` (or negative if `clip` is disabled) and
    reaches `w` at the peak. Use this when a value is best in the middle of a range,
    rather than as high or as low as possible.

    Examples:
        >>> PeakDesirabilityObjective(bounds=[0.0, 10.0], peak_position=7.0)
    """

    type: Literal["PeakDesirabilityObjective"] = "PeakDesirabilityObjective"
    log_shape_factor: float = Field(
        default=0.0,
        description="Logarithm of the shape factor for the increasing part: whether "
        "the interpolation between the lower bound and the peak is linear (=0), "
        "convex (>0) or concave (<0).",
    )
    log_shape_factor_decreasing: float = Field(
        default=0.0,
        description="Logarithm of the shape factor for the decreasing part: whether "
        "the interpolation between the peak and the upper bound is linear (=0), "
        "convex (>0) or concave (<0).",
    )  # often named log_t
    peak_position: float = Field(
        default=0.5,
        description="Position of the peak, where the desirability reaches its "
        "maximum. Must lie within `bounds`.",
    )  # often named T

    def to_description(self) -> str:
        raise NotImplementedError

    def call_numpy(
        self,
        x: np.ndarray,
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        y = np.zeros(x.shape)
        if self.clip:
            Incr = (x >= self.lower_bound) & (x <= self.peak_position)
            Decr = (x <= self.upper_bound) & (x > self.peak_position)
        else:
            Incr, Decr = x <= self.peak_position, x > self.peak_position

        s: float = np.exp(self.log_shape_factor)
        t: float = np.exp(self.log_shape_factor_decreasing)
        y[Incr] = np.power(
            np.divide(
                (x[Incr] - self.lower_bound), (self.peak_position - self.lower_bound)
            ),
            s,
        )
        y[Decr] = np.power(
            np.divide(
                (x[Decr] - self.upper_bound), (self.peak_position - self.upper_bound)
            ),
            t,
        )

        return y * self.w

    @pydantic.model_validator(mode="after")
    def validate_peak_position(self):
        bounds = self.bounds
        if self.peak_position < bounds[0] or self.peak_position > bounds[1]:
            raise ValueError(
                f"Peak position must be within bounds {bounds}, got {self.peak_position}"
            )
        return self


class InRangeDesirability(DesirabilityObjective):
    """A rectangular objective: desirability is one inside `bounds` and zero outside.

    Use when any value within the range is equally acceptable. The other desirability
    objectives grade values continuously; this one does not, so `clip` and the log
    shape factors have no effect on its shape.

    Examples:
        >>> InRangeDesirability(bounds=[2.0, 8.0])
    """

    type: Literal["InRangeDesirability"] = "InRangeDesirability"

    def to_description(self) -> str:
        raise NotImplementedError

    def call_numpy(
        self,
        x: np.ndarray,
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        y = np.zeros(x.shape)

        between = (x >= self.lower_bound) & (x <= self.upper_bound)
        y[between] = 1.0

        return y
