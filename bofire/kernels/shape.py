import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor


class WassersteinKernel(Kernel):
    has_lengthscale = True

    def __init__(self, squared: bool = False, **kwargs):
        super(WassersteinKernel, self).__init__(**kwargs)
        self.squared = squared

    def calc_wasserstein_distances(self, x1: Tensor, x2: Tensor):
        return (torch.cdist(x1, x2, p=1) / x1.shape[-1]).clamp_min(1e-15)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        dists = self.calc_wasserstein_distances(x1, x2)
        dists = dists / self.lengthscale
        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)
