from typing import List, Literal, Optional

import gpytorch
import torch

from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.priors.api import AnyPrior


class ContinuousKernel(Kernel):
    pass


class RBFKernel(ContinuousKernel):
    type: Literal["RBFKernel"] = "RBFKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> gpytorch.kernels.RBFKernel:
        return gpytorch.kernels.RBFKernel(
            batch_shape=batch_shape,
            ard_num_dims=len(active_dims) if self.ard else None,
            active_dims=active_dims,  # type: ignore
            lengthscale_prior=self.lengthscale_prior.to_gpytorch()
            if self.lengthscale_prior is not None
            else None,
        )


class MaternKernel(ContinuousKernel):
    type: Literal["MaternKernel"] = "MaternKernel"
    ard: bool = True
    nu: float = 2.5
    lengthscale_prior: Optional[AnyPrior] = None

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> gpytorch.kernels.MaternKernel:
        return gpytorch.kernels.MaternKernel(
            batch_shape=batch_shape,
            ard_num_dims=len(active_dims) if self.ard else None,
            active_dims=active_dims,
            nu=self.nu,
            lengthscale_prior=self.lengthscale_prior.to_gpytorch()
            if self.lengthscale_prior is not None
            else None,
        )


class LinearKernel(ContinuousKernel):
    type: Literal["LinearKernel"] = "LinearKernel"

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> gpytorch.kernels.LinearKernel:
        return gpytorch.kernels.LinearKernel(
            batch_shape=batch_shape, active_dims=active_dims
        )
