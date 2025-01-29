from abc import abstractmethod
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import pydantic

from bofire.data_models.objectives.identity import IdentityObjective


class DesirabilityObjective(IdentityObjective):
    """Abstract class for desirability objectives. Works as Identity Objective"""

    type: Literal["DesirabilityObjective"] = "DesirabilityObjective"  # type: ignore
    clip: bool = True

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

    Attributes:
        clip (bool): Whether to clip the values below/above the lower/upper bound, by
            default True.
        log_shape_factor (float): Logarithm of the shape factor:
            Whether the interpolation between the lower bound and the upper is linear (=0),
            convex (>0) or concave (<0) , by default 0.0.
        w (float): relative weight, by default = 1.
        bounds (tuple[float]): lower and upper bound of the desirability. Below
            bounds[0] the desirability is =0 (if clip=True) or <0 (if clip=False). Above
            bounds[1] the desirability is =1  (if clip=True) or >1 (if clip=False).
            Defaults to (0, 1).
    """

    type: Literal["IncreasingDesirabilityObjective"] = "IncreasingDesirabilityObjective"  # type: ignore
    log_shape_factor: float = 0.0

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

    Attributes:
        clip (bool): Whether to clip the values below/above the lower/upper bound, by
            default True.
        log_shape_factor (float): Logarithm of the shape factor:
            Whether the interpolation between the lower bound and the upper is linear (=0),
            convex (>0) or concave (<0) , by default 0.0.
        w (float): relative weight, by default = 1.
        bounds (tuple[float]): lower and upper bound of the desirability. Below
            bounds[0] the desirability is =1 (if clip=True) or >1 (if clip=False). Above
            bounds[1] the desirability is =0  (if clip=True) or <0 (if clip=False).
            Defaults to (0, 1).
    """

    type: Literal["DecreasingDesirabilityObjective"] = "DecreasingDesirabilityObjective"  # type: ignore
    log_shape_factor: float = 0.0

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

    Attributes:
        clip (bool): Whether to clip the values below/above the lower/upper bound, by
            default True.
        log_shape_factor (float): Logarithm of the shape factor for the increasing part:
            Whether the interpolation between the lower bound and the peak is linear (=0),
            convex (>1) or concave (<1) , by default 0.0.
        log_shape_factor_decreasing (float): Logarithm of the shape factor for the
            decreasing part. Whether the interpolation between the peak and the upper
            bound is linear (=0), convex (>0) or concave (<0), by default 0.0.
        peak_position (float): Position of the peak, by default 0.5.
        w (float): relative weight: desirability, when x=peak_position, by default = 1.
        bounds (tuple[float]): lower and upper bound of the desirability. Below
            bounds[0] the desirability is =0 (if clip=True) or <0 (if clip=False). Above
            bounds[1] the desirability is =0  (if clip=True) or <0 (if clip=False).
            Defaults to (0, 1).
    """

    type: Literal["PeakDesirabilityObjective"] = "PeakDesirabilityObjective"  # type: ignore
    log_shape_factor: float = 0.0
    log_shape_factor_decreasing: float = 0.0  # often named log_t
    peak_position: float = 0.5  # often named T

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
    type: Literal["InRangeDesirability"] = "InRangeDesirability"  # type: ignore

    def call_numpy(
        self,
        x: np.ndarray,
        x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        y = np.zeros(x.shape)

        between = (x >= self.lower_bound) & (x <= self.upper_bound)
        y[between] = 1.0

        return y
