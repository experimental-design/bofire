from abc import abstractmethod
from typing import List

import torch
from gpytorch.kernels import Kernel as GpytorchKernel

from bofire.data_models.base import BaseModel


class Kernel(BaseModel):
    type: str
    """
    A base class for creating kernel classes for machine learning and statistical modeling tasks.

    Attributes:
        type (str): The type of the kernel.

    Methods:
        to_gpytorch(batch_shape, ard_num_dims, active_dims): Convert the kernel object to a GpytorchKernel object.
    """

    @abstractmethod
    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        """
        Convert the kernel object to a GpytorchKernel object.

        Args:
            batch_shape (torch.Size): The shape of the input data batch.
            ard_num_dims (int): The number of dimensions for automatic relevance determination (ARD).
            active_dims (List[int]): The indices of the active dimensions.

        Returns:
            GpytorchKernel: A GpytorchKernel object.
        """

        pass
