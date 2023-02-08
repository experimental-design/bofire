import gpytorch.kernels
import pytest
import torch

from bofire.models.gps.kernels import LinearKernel, MaternKernel, RBFKernel
from bofire.models.gps.priors import GammaPrior


@pytest.mark.parametrize(
    "kernel, ard_num_dims, active_dims, expected_kernel",
    [
        (
            RBFKernel(
                ard=False, lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15)
            ),
            10,
            list(range(5)),
            gpytorch.kernels.RBFKernel,
        ),
        (
            RBFKernel(ard=False),
            10,
            list(range(5)),
            gpytorch.kernels.RBFKernel,
        ),
        (RBFKernel(ard=True), 10, list(range(5)), gpytorch.kernels.RBFKernel),
        (
            MaternKernel(
                ard=False, lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15)
            ),
            10,
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (MaternKernel(ard=False), 10, list(range(5)), gpytorch.kernels.MaternKernel),
        (MaternKernel(ard=True), 10, list(range(5)), gpytorch.kernels.MaternKernel),
        (
            MaternKernel(ard=False, nu=2.5),
            10,
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (
            MaternKernel(ard=True, nu=1.5),
            10,
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (LinearKernel(), 10, list(range(5)), gpytorch.kernels.LinearKernel),
    ],
)
def test_continuous_kernel(kernel, ard_num_dims, active_dims, expected_kernel):
    k = kernel.to_gpytorch(
        batch_shape=torch.Size(), ard_num_dims=ard_num_dims, active_dims=active_dims
    )
    assert isinstance(k, expected_kernel)
    if isinstance(kernel, LinearKernel):
        return
    if kernel.lengthscale_prior is not None:
        assert hasattr(k, "lengthscale_prior")
        assert isinstance(k.lengthscale_prior, gpytorch.priors.GammaPrior)
    else:
        assert hasattr(k, "lengthscale_prior") is False

    if kernel.ard is False:
        assert k.ard_num_dims is None
    else:
        assert k.ard_num_dims == len(active_dims)
    assert torch.eq(k.active_dims, torch.tensor(active_dims, dtype=torch.int64)).all()

    if isinstance(kernel, gpytorch.kernels.MaternKernel):
        assert kernel.nu == k.nu
