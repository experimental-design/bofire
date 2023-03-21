from typing import List, Literal, Optional

import gpytorch
import torch

from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.priors.api import AnyPrior


class ContinuousKernel(Kernel):
    pass


class RBFKernel(ContinuousKernel):
    """
    The RBFKernel class is a continuous kernel class that inherits from the ContinuousKernel class. It represents the radial basis function (RBF) kernel, which is a popular kernel function used in Gaussian process regression.

    Attributes:
    - type: A literal attribute representing the type of the kernel. It is always "RBFKernel".
    - ard: A boolean attribute representing whether the Automatic Relevance Determination (ARD) technique should be used to learn a separate lengthscale parameter for each input dimension. Default is True.
    - lengthscale_prior: An optional attribute representing the prior distribution on the lengthscale parameter of the kernel. If None, there is no prior distribution. Default is None.

    Methods:
    - to_gpytorch(batch_shape, ard_num_dims, active_dims): A method that converts the RBFKernel object to an RBFKernel object from the gpytorch library. It takes in the batch shape of the data, the number of ARD dimensions, and the indices of the active dimensions as inputs and returns an RBFKernel object.

    Note: The gpytorch library is a Gaussian process library for PyTorch that provides efficient and scalable Gaussian process inference with derivatives.
    """

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
    """A class representing a Matern kernel used for Gaussian process regression.

    The Matern kernel is a commonly used kernel in Gaussian process regression for modeling smooth functions. It is a subclass of ContinuousKernel.

    Attributes:
    type (Literal["MaternKernel"]): A string literal representing the type of kernel.
    ard (bool): A boolean indicating whether to use Automatic Relevance Determination (ARD) or not.
    nu (float): A float indicating the smoothness of the kernel. Must be greater than or equal to 0.5.
    lengthscale_prior (Optional[AnyPrior]): An optional prior distribution for the lengthscale parameter.

    Methods:
    to_gpytorch(batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]) -> gpytorch.kernels.MaternKernel:
    Converts the MaternKernel object to a gpytorch MaternKernel object with the given batch shape, number of dimensions
    for ARD, and active dimensions. Returns the converted gpytorch MaternKernel object.
    """

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
    """
    This is a class definition for a linear kernel derived from the ContinuousKernel base class. The LinearKernel class represents a kernel that computes the dot product between the input vectors.

    Attributes:
    type (Literal["LinearKernel"]): A string literal representing the type of kernel as LinearKernel.

    Methods:
    to_gpytorch(batch_shape, ard_num_dims, active_dims): A method that takes in the batch shape, the number of dimensions for automatic relevance determination (ARD), and a list of indices for active dimensions as input, and returns an instance of the LinearKernel class from the gpytorch.kernels module with the given batch shape and active dimensions.
    """

    type: Literal["LinearKernel"] = "LinearKernel"

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> gpytorch.kernels.LinearKernel:
        return gpytorch.kernels.LinearKernel(
            batch_shape=batch_shape, active_dims=active_dims
        )
