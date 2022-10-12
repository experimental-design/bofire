from typing import Dict, List, Optional, Tuple

import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.models.transforms.outcome import Standardize
from everest.domain.util import BaseModel
from everest.strategies.strategy import KernelEnum, ScalerEnum
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls import (ExactMarginalLogLikelihood,
                           LeaveOneOutPseudoLikelihood)
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


class ContKernelFactory(BaseModel):

    kernel: KernelEnum = KernelEnum.MATERN_25
    use_ard: bool = True
    active_dims: Optional[List[int]]

    def __call__(self):
        if self.kernel == KernelEnum.MATERN_25:
            return MaternKernel(
                nu=2.5,
                ard_num_dims=len(self.active_dims) if self.use_ard else None,
                active_dims=self.active_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        elif self.kernel == KernelEnum.MATERN_15:
            return MaternKernel(
                nu=1.5,
                ard_num_dims=len(self.active_dims) if self.use_ard else None,
                active_dims=self.active_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        elif self.kernel == KernelEnum.MATERN_05:
            return MaternKernel(
                nu=0.5,
                ard_num_dims=len(self.active_dims) if self.use_ard else None,
                active_dims=self.active_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        elif self.kernel == KernelEnum.RBF:
            return RBFKernel(
                ard_num_dims=len(self.active_dims) if self.use_ard else None,
                active_dims=self.active_dims,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )

    def to_mixedGP(self,batch_shape: torch.Size, ard_num_dims: int, active_dims: List[int]):
        if self.kernel == KernelEnum.MATERN_25:
            return MaternKernel(
                        nu=2.5,
                        batch_shape=batch_shape,
                        ard_num_dims=len(self.active_dims) if self.use_ard else None,
                        active_dims=self.active_dims,
                        lengthscale_constraint=GreaterThan(1e-04),
                    )
        elif self.kernel == KernelEnum.MATERN_15:
            return MaternKernel(
                        nu=1.5,
                        batch_shape=batch_shape,
                        ard_num_dims=len(self.active_dims) if self.use_ard else None,
                        active_dims=self.active_dims,
                        lengthscale_constraint=GreaterThan(1e-04),
                    )
        elif self.kernel == KernelEnum.MATERN_05:
            return MaternKernel(
                        nu=0.5,
                        batch_shape=batch_shape,
                        ard_num_dims=len(self.active_dims) if self.use_ard else None,
                        active_dims=self.active_dims,
                        lengthscale_constraint=GreaterThan(1e-04),
                    )
        else:
            return RBFKernel(
                batch_shape=batch_shape,
                ard_num_dims=len(self.active_dims) if self.use_ard else None,
                active_dims=self.active_dims,
                lengthscale_constraint=GreaterThan(1e-04),
            )

def get_dim_subsets(d:int, active_dims: List[int], cat_dims: List[int]):

    def check_indices(d, indices):
        if len(set(indices)) != len(indices):
            raise ValueError("Elements of `indices` list must be unique!")
        if any([i > d-1 for i in indices]):
            raise ValueError("Elements of `indices` have to be smaller than `d`!")
        if len(indices) > d:
            raise ValueError("Can provide at most `d` indices!")
        if any([i < 0 for i in indices]):
            raise ValueError("Elements of `indices` have to be smaller than `d`!")
        return indices

    if len(active_dims) == 0:
        raise ValueError("At least one active dim has to be provided!")

    # check validity of indices
    active_dims = check_indices(d, active_dims) 
    cat_dims = check_indices(d, cat_dims)

    # compute subsets
    ord_dims = sorted(set(range(d)) - set(cat_dims))
    ord_active_dims = sorted(
        set(active_dims) - set(cat_dims)
    )  # includes also descriptors
    cat_active_dims = sorted([i for i in cat_dims if i in active_dims])
    return ord_dims, ord_active_dims, cat_active_dims


def get_and_fit_model(
    train_X: Tensor,
    train_Y: Tensor,
    active_dims: List,
    cat_dims: List,
    scaler_name: ScalerEnum,
    kernel_name: KernelEnum,
    use_ard: bool = True,
    use_categorical_kernel: bool = True,
    cv: bool = False,
    #use_loocv_pseudo_likelihood: bool = False,
    bounds: Optional[Tensor] = None,
    maxiter: int = 15000
) -> SingleTaskGP:
    if train_X.dim() == 2:
        batch_shape = torch.Size()
    else:
        batch_shape = torch.Size([train_X.shape[0]])

    d = train_X.shape[-1]
    ord_dims, ord_active_dims, cat_active_dims = get_dim_subsets(d, active_dims, cat_dims)

    if len(cat_active_dims) != len(cat_dims):
        raise ValueError("Inactive categorical dimensions are not yet supported!")

    # first get the scaler
    if scaler_name == ScalerEnum.NORMALIZE:
        scaler = Normalize(d=train_X.shape[-1], indices=ord_dims if len(ord_dims) != d else None, bounds=bounds, batch_shape=batch_shape)
    elif scaler_name == ScalerEnum.STANDARDIZE:
        scaler = InputStandardize(d=train_X.shape[-1], indices=ord_dims if len(ord_dims) != d else None, batch_shape=batch_shape)
    else:
        raise ValueError("Scaler %s not implemented" % scaler_name)

    kernel_factory = ContKernelFactory(kernel=kernel_name, use_ard=use_ard, active_dims = ord_active_dims if use_categorical_kernel else active_dims)

    if (len(cat_dims) == 0) or (use_categorical_kernel == False):
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=ScaleKernel(kernel_factory(),outputscale_prior=GammaPrior(2.0,0.15)),
            outcome_transform=Standardize(m=1, batch_shape=batch_shape),
            input_transform=scaler,
        )
    else:
        model = MixedSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            cont_kernel_factory=kernel_factory.to_mixedGP,
            outcome_transform=Standardize(m=1, batch_shape=batch_shape),
            input_transform=scaler,
        )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if cv: mll.to(train_X)
    fit_gpytorch_model(mll,options = {"maxiter":maxiter})
    return model
