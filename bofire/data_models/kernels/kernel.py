from abc import abstractmethod
from typing import List

import torch
from gpytorch.kernels import Kernel as GpytorchKernel

from bofire.data_models.base import BaseModel


class Kernel(BaseModel):
    type: str

    @abstractmethod
    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        pass

    # TODO: can this be removed? circular import...
    # def __add__(self, other):
    #     return AdditiveKernel(kernels=[self, other])  # type: ignore

    # TODO: can this be removed? circular import...
    # def __mul__(self, other):
    #     return MultiplicativeKernel(kernels=[self, other])  # type: ignore
