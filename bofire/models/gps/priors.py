from abc import abstractmethod
from functools import partial
from typing import Literal, Union

import gpytorch.priors
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel, parse_obj_as
from pydantic.types import PositiveFloat


class Prior(BaseModel):
    """Abstract Prior class."""

    type: str

    @abstractmethod
    def to_gpytorch(self) -> gpytorch.priors.Prior:
        """Transform prior to a gpytorch prior.

        Returns:
            gpytorch.priors.Prior: Equivalent gpytorch prior object.
        """
        pass

    @staticmethod
    def from_dict(dict_: dict):
        """Parse object from dictionary.

        Args:
            dict_ (dict): Dictionary serialized prior class.

        Returns:
            AnyPrior: Instantiated prior class.
        """
        return parse_obj_as(AnyPrior, dict_)

    def plot_pdf(self, lower: float, upper: float):
        """Plot the probability density function of the prior with matplotlib.

        Args:
            lower (float): lower bound for computing the prior pdf.
            upper (float): upper bound for computing the prior pdf.

        Returns:
            fig, ax objects of the plot.
        """
        x = np.linspace(lower, upper, 1000)
        fig, ax = plt.subplots()
        ax.plot(
            x,
            np.exp(self.to_gpytorch().log_prob(torch.from_numpy(x)).numpy()),
            label="botorch",
        )
        return fig, ax


class GammaPrior(Prior):
    """Gamma prior based on the gamma distribution

    Attributes:
        concentration(PostiveFloat): concentration of the gamma distribution
        rate(PositiveFloat): rate of the gamma prior.
    """

    type: Literal["GammaPrior"] = "GammaPrior"
    concentration: PositiveFloat
    rate: PositiveFloat

    def to_gpytorch(self) -> gpytorch.priors.GammaPrior:
        return gpytorch.priors.GammaPrior(
            concentration=self.concentration, rate=self.rate
        )


class NormalPrior(Prior):
    """Normal prior based on the normal distribution

    Attributes:
        loc(float): mean/center of the normal distribution
        scale(PositiveFloat): width of the normal distribution
    """

    type: Literal["NormalPrior"] = "NormalPrior"
    loc: float
    scale: PositiveFloat

    def to_gpytorch(self) -> gpytorch.priors.NormalPrior:
        return gpytorch.priors.NormalPrior(loc=self.loc, scale=self.scale)


AnyPrior = Union[GammaPrior, NormalPrior]

# default priors of interest
# botorch defaults
botorch_lengthcale_prior = partial(GammaPrior, concentration=3.0, rate=6.0)
botorch_noise_prior = partial(GammaPrior, concentration=1.1, rate=0.05)
botorch_scale_prior = partial(GammaPrior, concentration=2.0, rate=0.15)

# mbo priors
# By default BoTorch places a highly informative prior on the kernel lengthscales,
# which easily leads to overfitting. Here we set a broader prior distribution for the
# lengthscale. The priors for the noise and signal variance are set more tightly.
mbo_lengthcale_prior = partial(GammaPrior, concentration=2.0, rate=0.2)
mbo_noise_prior = partial(GammaPrior, concentration=2.0, rate=4.0)
mbo_outputscale_prior = partial(GammaPrior, concentration=2.0, rate=4.0)
