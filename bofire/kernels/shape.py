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
        # to allow for ard now
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale

        print("x1 shape:", x1.shape, "x2 shape:", x2.shape)

        dists = self.calc_wasserstein_distances(x1_scaled, x2_scaled)

        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)


class ExactWassersteinKernel(Kernel):
    "This should already receive the area under the curves of x1 and x2 as inputs"

    # TODO: ARD support where x1, and x2 would have areas belonging to different ranges

    has_lengthscale = True

    def __init__(self, squared: bool = False, **kwargs):
        super(ExactWassersteinKernel, self).__init__(**kwargs)
        self.squared = squared

    def calc_distances(self, x1: Tensor, x2: Tensor):
        return (torch.cdist(x1, x2, p=1)).clamp_min(1e-15)

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

        # print("x1_scaled:", x1_scaled)
        # print("x2_scaled:", x2_scaled)

        dists = self.calc_distances(x1_scaled, x2_scaled)

        # print("dists:", dists)
        # dists = dists / self.lengthscale
        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)
