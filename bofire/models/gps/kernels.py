from abc import abstractmethod
from typing import List, Literal, Optional, Sequence, Union

import gpytorch.kernels
import torch
from gpytorch.kernels import Kernel as GpytorchKernel

from bofire.any.prior import AnyPrior
from bofire.domain.util import PydanticBaseModel


class BaseKernel(PydanticBaseModel):
    type: str

    @abstractmethod
    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        pass

    def __add__(self, other):
        return AdditiveKernel(kernels=[self, other])  # type: ignore

    def __mul__(self, other):
        return MultiplicativeKernel(kernels=[self, other])  # type: ignore


class ContinuousKernel(BaseKernel):
    type: Literal["ContinuousKernel"] = "ContinuousKernel"


class CategoricalKernel(BaseKernel):
    type: Literal["CategoricalKernel"] = "CategoricalKernel"


class HammondDistanceKernel(CategoricalKernel):
    type: Literal["HammondDistanceKernel"] = "HammondDistanceKernel"
    ard: bool = True

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> GpytorchKernel:
        raise NotImplementedError


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


class AdditiveKernel(BaseKernel):
    type: Literal["AdditiveKernel"] = "AdditiveKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            MaternKernel,
            LinearKernel,
            HammondDistanceKernel,
            "AdditiveKernel",
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ]
    type: Literal["AdditiveKernel"] = "AdditiveKernel"

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> gpytorch.kernels.AdditiveKernel:
        return gpytorch.kernels.AdditiveKernel(
            *[  # type: ignore
                k.to_gpytorch(
                    batch_shape=batch_shape,
                    ard_num_dims=ard_num_dims,
                    active_dims=active_dims,
                )
                for k in self.kernels
            ]
        )


class MultiplicativeKernel(BaseKernel):
    type: Literal["MultiplicativeKernel"] = "MultiplicativeKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            MaternKernel,
            LinearKernel,
            HammondDistanceKernel,
            "AdditiveKernel",
            "MultiplicativeKernel",
            "ScaleKernel",
        ]
    ]

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> gpytorch.kernels.AdditiveKernel:
        return gpytorch.kernels.ProductKernel(
            *[  # type: ignore
                k.to_gpytorch(
                    batch_shape=batch_shape,
                    ard_num_dims=ard_num_dims,
                    active_dims=active_dims,
                )
                for k in self.kernels
            ]
        )


class ScaleKernel(BaseKernel):
    type: Literal["ScaleKernel"] = "ScaleKernel"
    base_kernel: Union[
        RBFKernel,
        MaternKernel,
        LinearKernel,
        HammondDistanceKernel,
        AdditiveKernel,
        MultiplicativeKernel,
    ]
    outputscale_prior: Optional[AnyPrior] = None

    def to_gpytorch(
        self, batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]
    ) -> gpytorch.kernels.ScaleKernel:
        return gpytorch.kernels.ScaleKernel(
            base_kernel=self.base_kernel.to_gpytorch(
                batch_shape=batch_shape,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
            ),
            outputscale_prior=self.outputscale_prior.to_gpytorch()
            if self.outputscale_prior is not None
            else None,
        )


MultiplicativeKernel.update_forward_refs()
AdditiveKernel.update_forward_refs()
