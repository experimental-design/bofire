import gpytorch
import gpytorch.kernels
import pytest
import torch
from botorch.models.kernels.categorical import CategoricalKernel

import bofire
import bofire.kernels.api as kernels
import bofire.kernels.shape as shapeKernels
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    HammingDistanceKernel,
    InfiniteWidthBNNKernel,
    LinearKernel,
    MaternKernel,
    MultiplicativeKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
    WassersteinKernel,
)
from bofire.data_models.priors.api import THREESIX_SCALE_PRIOR, GammaPrior
from tests.bofire.data_models.specs.api import Spec


try:
    from botorch.models.kernels import InfiniteWidthBNNKernel as BNNKernel
except ImportError:
    BNN_AVAILABLE = False
else:
    BNN_AVAILABLE = True


EQUIVALENTS = {
    RBFKernel: gpytorch.kernels.RBFKernel,
    MaternKernel: gpytorch.kernels.MaternKernel,
    LinearKernel: gpytorch.kernels.LinearKernel,
    ScaleKernel: gpytorch.kernels.ScaleKernel,
    AdditiveKernel: gpytorch.kernels.AdditiveKernel,
    MultiplicativeKernel: gpytorch.kernels.ProductKernel,
    TanimotoKernel: bofire.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
    HammingDistanceKernel: CategoricalKernel,
    WassersteinKernel: shapeKernels.WassersteinKernel,
}


def test_map(kernel_spec: Spec):
    kernel = kernel_spec.cls(**kernel_spec.typed_spec())
    if isinstance(kernel, InfiniteWidthBNNKernel):
        return
    gkernel = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
    )
    assert isinstance(gkernel, EQUIVALENTS[kernel.__class__])


@pytest.mark.skipif(BNN_AVAILABLE is False, reason="requires latest botorch")
def test_map_infinite_width_bnn_kernel():
    kernel = InfiniteWidthBNNKernel(depth=3)
    gkernel = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        ard_num_dims=10,
    )
    assert isinstance(gkernel, BNNKernel)


def test_map_scale_kernel():
    kernel = ScaleKernel(
        base_kernel=RBFKernel(), outputscale_prior=THREESIX_SCALE_PRIOR()
    )
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
    )
    assert hasattr(k, "outputscale_prior")
    assert isinstance(k.outputscale_prior, gpytorch.priors.GammaPrior)
    kernel = ScaleKernel(base_kernel=RBFKernel())
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
    )
    assert hasattr(k, "outputscale_prior") is False


def test_map_polynomial_kernel():
    kernel = PolynomialKernel(power=2, offset_prior=THREESIX_SCALE_PRIOR())
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
    )
    assert hasattr(k, "offset_prior")
    assert isinstance(k.offset_prior, gpytorch.priors.GammaPrior)
    kernel = PolynomialKernel(power=2)
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
    )
    assert hasattr(k, "offset_prior") is False


@pytest.mark.parametrize(
    "kernel, ard_num_dims, active_dims, expected_kernel",
    [
        (
            RBFKernel(
                ard=False,
                lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
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
                ard=False,
                lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
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
def test_map_continuous_kernel(kernel, ard_num_dims, active_dims, expected_kernel):
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
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


@pytest.mark.parametrize(
    "kernel, ard_num_dims, active_dims, expected_kernel",
    [
        (
            TanimotoKernel(ard=False),
            10,
            list(range(5)),
            bofire.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
        ),
        (
            TanimotoKernel(ard=True),
            10,
            list(range(5)),
            bofire.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
        ),
    ],
)
def test_map_molecular_kernel(kernel, ard_num_dims, active_dims, expected_kernel):
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
    )
    assert isinstance(k, expected_kernel)

    if kernel.ard is False:
        assert k.ard_num_dims is None
    else:
        assert k.ard_num_dims == len(active_dims)
    assert torch.eq(k.active_dims, torch.tensor(active_dims, dtype=torch.int64)).all()


def test_map_wasserstein_kernel():
    kernel = WassersteinKernel(
        squared=False,
        lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
    )
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
    )
    assert isinstance(k, shapeKernels.WassersteinKernel)
    assert hasattr(k, "lengthscale_prior")
    assert isinstance(k.lengthscale_prior, gpytorch.priors.GammaPrior)
    assert k.squared is False
    kernel = WassersteinKernel(squared=True)
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
    )
    assert k.squared is True
    assert hasattr(k, "lengthscale_prior") is False
