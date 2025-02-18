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
        outputscale_prior: Optional[gpytorch.priors.Prior] = None,
        outputscale_constraint: Optional[gpytorch.constraints.Interval] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.kernels = kernels
        self.max_degree = max_degree

        # each item corresponds to a different interaction term
        # and contains the indices of the kernels that interact
        self.indices: list[tuple[int, ...]] = [
            idx
            for n in range(1, self.max_degree + 1)
            for idx in (
                itertools.combinations_with_replacement(range(len(kernels)), n)
                if include_self_interactions
                else itertools.combinations(range(len(kernels)), n)
            )
        ]

        outputscale = (
            torch.zeros(*self.batch_shape, len(self.indices) + 1)
            if len(self.batch_shape)
            else torch.zeros(len(self.indices) + 1)
        )
        self.register_parameter(
            name="raw_outputscale", parameter=torch.nn.Parameter(outputscale)
        )
        if outputscale_prior is not None:
            if not isinstance(outputscale_prior, gpytorch.priors.Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(outputscale_prior).__name__
                )
            self.register_prior(
                "outputscale_prior",
                outputscale_prior,
                self._outputscale_param,
                self._outputscale_closure,
            )
        if outputscale_constraint is None:
            outputscale_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_outputscale", outputscale_constraint)

    def _outputscale_param(self, m):
        return m.outputscale

    def _outputscale_closure(self, m, v):
        m._set_outputscale(v)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(
            raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value)
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        base_kernels = torch.stack(
            [
                k(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch).to_dense()
                for k in self.kernels
            ],
            dim=0,
        )

        os = self.outputscale
        result = os[0] * torch.ones_like(base_kernels[0])  # constant term
        i = 1
        for idx in self.indices:
            # compute each interaction term between specified kernels,
            # that is, the product of the individual kernels scaled
            # by the outputscale of the interaction
            result += os[i] * base_kernels[idx, ...].prod(dim=0)
            i += 1

        return result
