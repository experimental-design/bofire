import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor


class WassersteinKernel(Kernel):
    has_lengthscale = True

    def __init__(self, squared: bool = False, **kwargs):
        super(WassersteinKernel, self).__init__(**kwargs)
        self.squared = squared

    def calc_wasserstein_distances(self, x1: Tensor, x2: Tensor, norm: Tensor):
        return (torch.cdist(x1, x2, p=1) / norm).clamp_min(1e-15)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        # to allow for ard now
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale

        # check if lengthscale is a vector or a scalar
        if self.lengthscale.numel() == 1:
            norm = x1.shape[-1]
        else:
            # sum lengthscales this did not work.
            norm = x1.shape[-1]  # torch.sum(self.lengthscale)

        dists = self.calc_wasserstein_distances(x1_scaled, x2_scaled, norm)

        # dists = dists / self.lengthscale
        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)
