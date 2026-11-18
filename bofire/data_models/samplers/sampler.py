import warnings
from typing import Any, Dict, Literal, Optional

import numpy as np
from pydantic import Field
from scipy.stats.qmc import LatinHypercube, Sobol

from bofire.data_models.base import BaseModel
from bofire.data_models.enum import SamplingMethodEnum


class Sampler(BaseModel):
    """Method that spreads sample points over the inputs, e.g. for an initial design.

    Categorical and discrete inputs take their values from equally wide bins of the
    sampled unit interval, so the samplers spread those values evenly too.
    """

    type: Any

    def unit_samples(self, n_dims: int, n: int, seed: Optional[int]) -> np.ndarray:
        """Draws `n` points from the unit cube of dimension `n_dims`.

        Args:
            n_dims: Number of dimensions.
            n: Number of points.
            seed: Seed for reproducible points; random if not provided.

        Returns:
            An array of shape `(n, n_dims)` with values in [0, 1].
        """
        return np.random.default_rng(seed).uniform(size=(n, n_dims))


class UniformSampler(Sampler):
    """Draws every input independently and uniformly at random.

    The simplest choice; with few points, the samples can cluster and leave gaps.

    Examples:
        >>> UniformSampler()
    """

    type: Literal["UniformSampler"] = "UniformSampler"


class _QuasiRandomSampler(Sampler):
    """Spreads the points more evenly than independent random draws, by a
    low-discrepancy design on the unit cube."""

    scramble: bool = Field(
        default=True,
        description="Whether the design is randomized; without scrambling, the same "
        "points are drawn every time.",
    )
    optimization: Optional[Literal["random-cd", "lloyd"]] = Field(
        default=None,
        description="Optional post-processing that spreads the points even more "
        "evenly: 'random-cd' lowers their centered discrepancy, 'lloyd' moves them "
        "apart by Lloyd-Max iterations. If not provided, none.",
    )


class SobolSampler(_QuasiRandomSampler):
    """Draws points of a Sobol sequence, a quasi-random design that covers the
    domain evenly.

    The coverage is best when the number of points is a power of two.

    Examples:
        >>> SobolSampler(scramble=True)
    """

    type: Literal["SobolSampler"] = "SobolSampler"

    def unit_samples(self, n_dims: int, n: int, seed: Optional[int]) -> np.ndarray:
        with warnings.catch_warnings():
            # scipy warns if n is not a power of two
            warnings.simplefilter("ignore")
            return Sobol(
                n_dims,
                scramble=self.scramble,
                optimization=self.optimization,
                seed=seed,
            ).random(n)


class LatinHypercubeSampler(_QuasiRandomSampler):
    """Draws a Latin hypercube design: the range of every input is split into as
    many intervals as there are points, and every interval holds exactly one point.

    Examples:
        >>> LatinHypercubeSampler(strength=2)
    """

    type: Literal["LatinHypercubeSampler"] = "LatinHypercubeSampler"
    strength: Literal[1, 2] = Field(
        default=1,
        description="1 for a plain Latin hypercube; 2 also spreads the points evenly "
        "over every pair of inputs, but needs the number of points to be the square "
        "of a prime and at most that prime plus one inputs.",
    )

    def unit_samples(self, n_dims: int, n: int, seed: Optional[int]) -> np.ndarray:
        return LatinHypercube(
            n_dims,
            scramble=self.scramble,
            strength=self.strength,
            optimization=self.optimization,
            seed=seed,
        ).random(n)


def sampler_from_method(
    method: SamplingMethodEnum, sampler_kwargs: Optional[Dict[str, Any]] = None
) -> Sampler:
    """Returns the sampler of a sampling method and its options.

    Args:
        method: The sampling method.
        sampler_kwargs: Options of the sampler, e.g. `{"scramble": False}`.

    Raises:
        ValidationError: if an option is not one of the sampler's fields.

    Returns:
        The sampler.
    """
    sampler_cls = {
        SamplingMethodEnum.UNIFORM: UniformSampler,
        SamplingMethodEnum.SOBOL: SobolSampler,
        SamplingMethodEnum.LHS: LatinHypercubeSampler,
    }[SamplingMethodEnum(method)]
    return sampler_cls(**(sampler_kwargs or {}))
