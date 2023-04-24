from typing import List, Literal

import torch
from gpytorch.kernels import Kernel as GpytorchKernel

from bofire.data_models.kernels.kernel import Kernel


class CategoricalKernel(Kernel):
    pass


class HammondDistanceKernel(CategoricalKernel):
    type: Literal["HammondDistanceKernel"] = "HammondDistanceKernel"
    ard: bool = True

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        raise NotImplementedError
