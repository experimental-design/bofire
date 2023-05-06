from typing import List, Literal

import torch
from gpytorch.kernels import Kernel as GpytorchKernel
from botorch.models.kernels.categorical import CategoricalKernel as BoTorchCategoricalKernel

from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel as GaucheTanimotoKernel


class CategoricalKernel(Kernel):
    pass


class HammingDistanceKernel(CategoricalKernel):
    type: Literal["HammingDistanceKernel"] = "HammingDistanceKernel"
    ard: bool = True

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        return BoTorchCategoricalKernel(
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims if self.ard else None,
            active_dims=active_dims,
        )

class TanimotoKernel(CategoricalKernel):
    type: Literal["TanimotoKernel"] = "TanimotoKernel"

    def to_gpytorch(
        self, batch_shape: torch.Size,
            ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        return GaucheTanimotoKernel(
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
        )
