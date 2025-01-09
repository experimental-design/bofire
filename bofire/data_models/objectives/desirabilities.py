from typing import Literal, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd

import torch
from torch import Tensor

from bofire.data_models.objectives.identity import IdentityObjective


class _SeriesNumpyCallable:
    """Helper class to call numpy and torch functions with series or numpy arrays. matches __call__
    signature of objectives."""
    def __call__(self, x: Union[pd.Series, np.ndarray], x_adapt) -> Union[pd.Series, np.ndarray]:

        convert_to_series = False
        if isinstance(x, pd.Series):
            convert_to_series = True
            name = x.name
            x = x.values

        y = self.call_numpy(x)

        if convert_to_series:
            return pd.Series(y, name=name)

        return y

    def call_numpy(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def torch_callable_factory(self, idx: int, x_adapt: Tensor) -> callable:
        raise NotImplementedError()

class DesirabilityObjective(IdentityObjective):
    """Abstract class for desirability objectives."""
    pass

class IncreasingDesirabilityObjective(_SeriesNumpyCallable, DesirabilityObjective):
    """An objective returning a reward the scaled identity, but trimmed at the bounds:

        d = ((x - lower_bound) / (upper_bound - lower_bound))^t

    if clip is True, the reward is zero for x < lower_bound and one for x > upper_bound.

    where:

        t = exp(log_shape_factor)

    Note, that with clipping the reward is always between zero and one.

    Attributes:
        log_shape_factor (float): float determining the shape of the desirability function
        clip (bool): bool determining if the reward is clipped at the bounds
    """

    log_shape_factor: float = 0.
    clip: bool = True

    def call_numpy(
            self,
            x: np.ndarray,
            x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:

        y = np.zeros(x.shape)
        if self.clip:
            y[x < self.lower_bound] = 0.
            y[x > self.upper_bound] = 1.
            between = (x >= self.lower_bound) & (x <= self.upper_bound)
        else:
            between = np.full(x.shape, True)

        t = np.exp(self.log_shape_factor)

        y[between] = np.power((x[between] - self.lower_bound) / (self.upper_bound - self.lower_bound), t)

        return y

    def torch_callable_factory(self, idx: int, x_adapt: Tensor) -> callable:

        def objective(x: Tensor, *args) -> Tensor:
            x = x[..., idx]

            y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            if self.clip:
                y[x < self.lower_bound] = 0.
                y[x > self.upper_bound] = 1.
                between = (x >= self.lower_bound) & (x <= self.upper_bound)
            else:
                between = torch.full(x.shape, True, dtype=torch.bool, device=x.device)

            t: float = np.exp(self.log_shape_factor)

            y[between] = torch.pow((x[between] - self.lower_bound) / (self.upper_bound - self.lower_bound), t)
            return y

        return objective




class DecreasingDesirabilityObjective(_SeriesNumpyCallable, DesirabilityObjective):
    """An objective returning a reward the negative, shifted scaled identity, but trimmed at the bounds:

        d = ((upper_bound - x) / (upper_bound - lower_bound))^t

    if clip is True, the reward is one for x < lower_bound and zero for x > upper_bound.

    where:

        t = exp(log_shape_factor)

    Note, that with clipping the reward is always between zero and one.

    Attributes:
        log_shape_factor (float): float determining the shape of the desirability function
        clip (bool): bool determining if the reward is clipped at the bounds
    """

    log_shape_factor: float = 0.
    clip: bool = True

    def call_numpy(
            self,
            x: np.ndarray,
            x_adapt: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:

        y = np.zeros(x.shape)
        if self.clip:
            y[x < self.lower_bound] = 1.
            y[x > self.upper_bound] = 0.
            between = (x >= self.lower_bound) & (x <= self.upper_bound)
        else:
            between = np.full(x.shape, True)

        t = np.exp(self.log_shape_factor)

        y[between] = np.power((self.upper_bound - x[between]) / (self.upper_bound - self.lower_bound), t)

        return y

    def torch_callable_factory(self, idx: int, x_adapt: Tensor) -> callable:
        def objective(x: Tensor, *args) -> Tensor:
            x = x[..., idx]

            y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            if self.clip:
                y[x < self.lower_bound] = 1.
                y[x > self.upper_bound] = 0.
                between = (x >= self.lower_bound) & (x <= self.upper_bound)
            else:
                between = torch.full(x.shape, True, dtype=torch.bool, device=x.device)

            t: float = np.exp(self.log_shape_factor)
            y[between] = torch.pow((self.upper_bound - x[between]) / (self.upper_bound - self.lower_bound), t)
            return y

        return objective

class PeakDesirabilityObjective(_SeriesNumpyCallable, DesirabilityObjective):
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
            bound is linear (=0), convex (>1) or concave (<1), by default 0.0.
        peak_position (float): Position of the peak, by default 0.5.
        w (float): relative weight: desirability, when x=peak_position, by default = 1.
        bounds (tuple[float]): lower and upper bound of the desirability. Below
            bounds[0] the desirability is =0 (if clip=True) or <0 (if clip=False). Above
            bounds[1] the desirability is =0  (if clip=True) or <0 (if clip=False).
            Defaults to (0, 1).
    """

    clip: bool = True
    log_shape_factor: float = 0.0  # often named log_s
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

    def torch_callable_factory(self, idx: int, x_adapt: Tensor) -> Callable:
        def objective(x: Tensor, *args) -> Tensor:
            x = x[..., idx]
            y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)

            if self.clip:
                Incr = (x >= self.lower_bound) & (x <= self.peak_position)
                Decr = (x <= self.upper_bound) & (x > self.peak_position)
            else:
                Incr, Decr = x <= self.peak_position, x > self.peak_position

            s: float = np.exp(self.log_shape_factor)
            t: float = np.exp(self.log_shape_factor_decreasing)
            y[Incr] = torch.pow(
                torch.divide(
                    (x[Incr] - self.lower_bound),
                    (self.peak_position - self.lower_bound),
                ),
                s,
            )
            y[Decr] = torch.pow(
                torch.divide(
                    (x[Decr] - self.upper_bound),
                    (self.peak_position - self.upper_bound),
                ),
                t,
            )
            return y * self.w

        return objective