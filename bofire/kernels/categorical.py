import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor


class HammingKernelWithOneHots(Kernel):
    has_lengthscale = True

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        delta = (x1.unsqueeze(-2) - x2.unsqueeze(-3))**2
        dists = delta / self.lengthscale.unsqueeze(-2)
        if last_dim_is_batch:
            dists = dists.transpose(-3, -1)

        dists = dists.sum(-1) / 2
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res
