import itertools
from typing import Any, Optional

import gpytorch
import gpytorch.constraints
import torch
from torch import Tensor


class PolynomialFeatureInteractionKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        kernels: list[gpytorch.kernels.Kernel],
        max_degree: int,
        include_self_interactions: bool,
        lengthscale_prior: Optional[gpytorch.priors.Prior] = None,
        lengthscale_constraint: Optional[gpytorch.constraints.Interval] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.kernels = kernels
        self.max_degree = max_degree
        self.indices = [
            list(
                itertools.combinations_with_replacement(range(len(kernels)), n)
                if include_self_interactions
                else itertools.combinations(range(len(kernels)), n)
            )
            for n in range(1, self.max_degree + 1)
        ]

        n = sum(len(idx) for idx in self.indices)
        lengthscale = (
            torch.zeros(*self.batch_shape, n)
            if len(self.batch_shape)
            else torch.zeros(n)
        )
        self.register_parameter(
            name="raw_lengthscale", parameter=torch.nn.Parameter(lengthscale)
        )
        if lengthscale_prior is not None:
            if not isinstance(lengthscale_prior, gpytorch.priors.Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(lengthscale_prior).__name__
                )
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                self._lengthscale_param,
                self._lengthscale_closure,
            )
        if lengthscale_constraint is None:
            lengthscale_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_lengthscale", lengthscale_constraint)

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    def _set_lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(
            raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value)
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        ks = torch.stack(
            [
                k(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch).to_dense()
                for k in self.kernels
            ],
            dim=0,
        )

        os = self.lengthscale
        rr = torch.zeros_like(ks[0])
        i = 0
        for idx_n in self.indices:
            for kk in idx_n:
                rr += os[i] * ks[kk, ...].prod(dim=0)
                i += 1

        return rr
