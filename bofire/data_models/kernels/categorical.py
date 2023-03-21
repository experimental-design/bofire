from typing import List, Literal

import torch
from gpytorch.kernels import Kernel as GpytorchKernel

from bofire.data_models.kernels.kernel import Kernel


class CategoricalKernel(Kernel):
    pass


class HammondDistanceKernel(CategoricalKernel):
    """
    A kernel for computing the Hammond distance between categorical variables.
    This class inherits from the CategoricalKernel class and implements the
    to_gpytorch method which converts the kernel to a GpytorchKernel object.

    Attributes:
    type (Literal[str]): A literal string representing the type of kernel as
    "HammondDistanceKernel".
    ard (bool): A boolean value indicating whether to use Automatic Relevance
    Determination (ARD) for different input dimensions. Default is True.

    Methods:
    to_gpytorch(batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int])
    : Raises a NotImplementedError since the conversion to GpytorchKernel
    object has not been implemented yet.
    """

    type: Literal["HammondDistanceKernel"] = "HammondDistanceKernel"
    ard: bool = True

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        raise NotImplementedError
