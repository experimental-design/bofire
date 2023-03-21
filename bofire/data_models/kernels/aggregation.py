from typing import List, Literal, Optional, Sequence, Union

import gpytorch
import torch

from bofire.data_models.kernels.categorical import HammondDistanceKernel
from bofire.data_models.kernels.continuous import LinearKernel, MaternKernel, RBFKernel
from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.priors.api import AnyPrior


class AdditiveKernel(Kernel):
    """
    A class representing an additive kernel in Gaussian process regression.

    Attributes:
        type (Literal["AdditiveKernel"]): A string literal indicating that this is an additive kernel.
        kernels (Sequence[Union[RBFKernel, MaternKernel, LinearKernel, HammondDistanceKernel, AdditiveKernel, MultiplicativeKernel, ScaleKernel]]):
            A sequence of kernel objects that will be added together.
    """

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


class MultiplicativeKernel(Kernel):
    """
    A class representing a multiplicative kernel in Gaussian process regression.

    Attributes:
        type (Literal["MultiplicativeKernel"]): A string literal indicating that this is a multiplicative kernel.
        kernels (Sequence[Union[RBFKernel, MaternKernel, LinearKernel, HammondDistanceKernel, AdditiveKernel, MultiplicativeKernel, ScaleKernel]]):
            A sequence of kernel objects that will be added together.
    """

    type: Literal["MultiplicativeKernel"] = "MultiplicativeKernel"
    kernels: Sequence[
        Union[
            RBFKernel,
            MaternKernel,
            LinearKernel,
            HammondDistanceKernel,
            AdditiveKernel,
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


class ScaleKernel(Kernel):
    """
    A kernel class that scales another base kernel by a fixed factor, resulting in a new kernel that can be used in Gaussian Process models.

    Attributes:
    type (Literal["ScaleKernel"]): A string literal indicating the type of the kernel as ScaleKernel.
    base_kernel (Union[RBFKernel, MaternKernel, LinearKernel, HammondDistanceKernel, AdditiveKernel, MultiplicativeKernel, ScaleKernel]): The base kernel to be scaled. Can be an instance of any of the supported kernel classes or another instance of ScaleKernel.
    outputscale_prior (Optional[AnyPrior]): An optional prior on the output scale of the kernel.
    """

    type: Literal["ScaleKernel"] = "ScaleKernel"
    base_kernel: Union[
        RBFKernel,
        MaternKernel,
        LinearKernel,
        HammondDistanceKernel,
        AdditiveKernel,
        MultiplicativeKernel,
        "ScaleKernel",
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


AdditiveKernel.update_forward_refs()
MultiplicativeKernel.update_forward_refs()
