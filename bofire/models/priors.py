from abc import abstractmethod
from typing import Literal, Union

import gpytorch.priors
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel, parse_obj_as
from pydantic.types import PositiveFloat


class Prior(BaseModel):

    type: str

    @abstractmethod
    def to_gpytorch(self) -> gpytorch.priors.Prior:
        pass

    @staticmethod
    def from_dict(dict_: dict):
        return parse_obj_as(AnyPrior, dict_)

    def plot_pdf(self, lower: float, upper: float):
        x = np.linspace(lower, upper, 1000)
        fig, ax = plt.subplots()
        ax.plot(
            x,
            np.exp(self.to_gpytorch().log_prob(torch.from_numpy(x)).numpy()),
            label="botorch",
        )
        return fig, ax


class GammaPrior(Prior):

    type: Literal["GammaPrior"] = "GammaPrior"
    concentration: PositiveFloat
    rate: PositiveFloat

    def to_gpytorch(self) -> gpytorch.priors.GammaPrior:
        return gpytorch.priors.GammaPrior(
            concentration=self.concentration, rate=self.rate
        )


class NormalPrior(Prior):

    type: Literal["NormalPrior"] = "NormalPrior"
    loc: float
    scale: PositiveFloat

    def to_gpytorch(self) -> gpytorch.priors.NormalPrior:
        return gpytorch.priors.NormalPrior(loc=self.loc, scale=self.scale)


AnyPrior = Union[GammaPrior, NormalPrior]

# default priors of interest
# botorch defaults
botorch_lengthcale_prior = GammaPrior(concentration=3.0, rate=6.0)
# botorch_noise_prior = GammaPrior(concentration=, rate=)
# botorch_scale_prior = GammaPrior(concentration=, rate=)

# mbo priors
# By default BoTorch places a highly informative prior on the kernel lengthscales,
# which easily leads to overfitting. Here we set a broader prior distribution for the
# lengthscale. The priors for the noise and signal variance are set more tightly.
mbo_lengthcale_prior = GammaPrior(concentration=2.0, rate=0.2)
mbo_noise_prior = GammaPrior(concentration=2.0, rate=4.0)
mbo_outputscale_prior = GammaPrior(concentration=2.0, rate=4.0)
