from abc import abstractmethod

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

from bofire.data_models.base import BaseModel


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

    # TODO: move this to a general plot component
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
