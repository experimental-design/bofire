import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from scipy.stats.qmc import LatinHypercube, Sobol

from bofire.data_models.base import BaseModel
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
    Input,
)


if TYPE_CHECKING:
    from bofire.data_models.domain.api import Inputs


class Sampler(BaseModel):
    """Method that spreads sample points over the inputs, e.g. for an initial design.

    Samplers are immutable: their options can't be changed after construction.
    """

    model_config = ConfigDict(frozen=True)

    type: Any

    def sample(self, inputs: "Inputs", n: int, seed: Optional[int]) -> pd.DataFrame:
        """Draws `n` samples of the inputs; fixed inputs keep their value.

        Args:
            inputs: The inputs to sample.
            n: Number of samples.
            seed: Seed for reproducible samples; random if not provided.

        Returns:
            A dataframe with one column per input.
        """
        raise NotImplementedError


class UniformSampler(Sampler):
    """Draws every input independently and uniformly at random.

    The simplest choice; with few points, the samples can cluster and leave gaps.

    Examples:
        >>> UniformSampler()
    """

    type: Literal["UniformSampler"] = "UniformSampler"

    def sample(self, inputs: "Inputs", n: int, seed: Optional[int]) -> pd.DataFrame:
        # every input gets a seed of its own, so inputs with the same bounds are
        # not sampled identically
        rng = np.random.default_rng(seed=seed)
        return pd.concat(
            [
                feat.sample(n, seed=int(rng.integers(1, 1000000)))
                for feat in inputs.get(Input)
            ],
            axis=1,
        )


class _QuasiRandomSampler(Sampler):
    """Spreads the points more evenly than independent random draws, by a
    low-discrepancy design on the unit cube.

    Categorical and discrete inputs take their values from equally wide bins of the
    unit interval, so their values are spread evenly too.
    """

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

    def _unit_samples(self, n_dims: int, n: int, seed: Optional[int]) -> np.ndarray:
        raise NotImplementedError

    def sample(self, inputs: "Inputs", n: int, seed: Optional[int]) -> pd.DataFrame:
        free_features = inputs.get_free()
        X = self._unit_samples(len(free_features), n, seed)

        res = []
        for i, feat in enumerate(free_features):
            if isinstance(feat, ContinuousInput):
                x = feat.from_unit_range(X[:, i])
            elif isinstance(feat, (DiscreteInput, CategoricalInput)):
                levels = (
                    feat.values
                    if isinstance(feat, DiscreteInput)
                    else feat.get_allowed_categories()
                )
                bins = np.linspace(0, 1, len(levels) + 1)
                idx = np.digitize(X[:, i], bins) - 1
                x = np.array(levels)[idx]
            else:
                raise ValueError(
                    f"Unknown input feature with key {feat.key} of type {feat.type}",
                )
            res.append(pd.Series(x, name=feat.key))

        samples = pd.concat(res, axis=1)
        for feat in inputs.get_fixed():
            val = feat.fixed_value()
            assert val is not None
            samples[feat.key] = val[0]
        return samples


class SobolSampler(_QuasiRandomSampler):
    """Draws points of a Sobol sequence, a quasi-random design that covers the
    domain evenly.

    The coverage is best when the number of points is a power of two.

    Examples:
        >>> SobolSampler(scramble=True)
    """

    type: Literal["SobolSampler"] = "SobolSampler"

    def _unit_samples(self, n_dims: int, n: int, seed: Optional[int]) -> np.ndarray:
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

    def _unit_samples(self, n_dims: int, n: int, seed: Optional[int]) -> np.ndarray:
        return LatinHypercube(
            n_dims,
            scramble=self.scramble,
            strength=self.strength,
            optimization=self.optimization,
            seed=seed,
        ).random(n)
