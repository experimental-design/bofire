from typing import List

import gpytorch
import gpytorch.kernels
import pytest
import torch
from pydantic import parse_obj_as

import bofire
import bofire.kernels.api as kernels
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    AnyKernel,
    LinearKernel,
    MaternKernel,
    MultiplicativeKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
)
from bofire.data_models.priors.api import BOTORCH_SCALE_PRIOR, GammaPrior


def get_invalids(valid: dict) -> List[dict]:
    return [
        {k: v for k, v in valid.items() if k != k_}
        for k_ in valid.keys()
        if k_ != "type"
    ]


EQUIVALENTS = {
    RBFKernel: gpytorch.kernels.RBFKernel,
    MaternKernel: gpytorch.kernels.MaternKernel,
    LinearKernel: gpytorch.kernels.LinearKernel,
    ScaleKernel: gpytorch.kernels.ScaleKernel,
    AdditiveKernel: gpytorch.kernels.AdditiveKernel,
    MultiplicativeKernel: gpytorch.kernels.ProductKernel,
    TanimotoKernel: bofire.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
}

VALID_RBF_SPEC = {
    "type": "RBFKernel",
    "ard": True,
}
VALID_MATERN_SPEC = {
    "type": "MaternKernel",
    "ard": True,
    "nu": 2.5,
}
VALID_LINEAR_SPEC = {"type": "LinearKernel"}

VALID_SCALE_SPEC = {"type": "ScaleKernel", "base_kernel": RBFKernel()}

VALID_ADDITIVE_SPEC = {"type": "AdditiveKernel", "kernels": [RBFKernel(), RBFKernel()]}

VALID_MULTIPLICATIVE_SPEC = {
    "type": "MultiplicativeKernel",
    "kernels": [RBFKernel(), RBFKernel()],
}

VALID_TANIMOTO_SPEC = {
    "type": "TanimotoKernel",
    "ard": True,
}

KERNEL_SPECS = {
    RBFKernel: {
        "valids": [
            VALID_RBF_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_RBF_SPEC),
        ],
    },
    MaternKernel: {
        "valids": [
            VALID_MATERN_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_MATERN_SPEC),
        ],
    },
    LinearKernel: {
        "valids": [
            VALID_LINEAR_SPEC,
        ],
        "invalids": [
            *get_invalids(VALID_LINEAR_SPEC),
        ],
    },
    ScaleKernel: {
        "valids": [VALID_SCALE_SPEC],
        "invalids": [
            *get_invalids(VALID_SCALE_SPEC),
        ],
    },
    MultiplicativeKernel: {
        "valids": [VALID_MULTIPLICATIVE_SPEC],
        "invalids": [
            *get_invalids(VALID_SCALE_SPEC),
        ],
    },
    AdditiveKernel: {
        "valids": [VALID_ADDITIVE_SPEC],
        "invalids": [
            *get_invalids(VALID_SCALE_SPEC),
        ],
    },
    TanimotoKernel: {
        "valids": [VALID_TANIMOTO_SPEC],
        "invalids": [
            *get_invalids(VALID_TANIMOTO_SPEC),
        ],
    },
}


@pytest.mark.parametrize(
    "cls, spec",
    [(cls, valid) for cls, data in KERNEL_SPECS.items() for valid in data["valids"]],
)
def test_valid_kernel_specs(cls, spec):
    res = cls(**spec)
    assert isinstance(res, cls)
    assert isinstance(res.__str__(), str)
    gkernel = kernels.map(
        res, batch_shape=torch.Size(), ard_num_dims=10, active_dims=list(range(5))
    )
    assert isinstance(gkernel, EQUIVALENTS[cls])
    res2 = parse_obj_as(AnyKernel, res.dict())
    assert res == res2


def test_scale_kernel():
    kernel = ScaleKernel(
        base_kernel=RBFKernel(), outputscale_prior=BOTORCH_SCALE_PRIOR()
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


def test_poly_kernel():
    kernel = PolynomialKernel(power=2, offset_prior=BOTORCH_SCALE_PRIOR())
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
def test_molecular_kernel(kernel, ard_num_dims, active_dims, expected_kernel):
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
