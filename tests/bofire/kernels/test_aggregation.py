import itertools

import pytest
import torch
from gpytorch.kernels import (
    ConstantKernel,
    CosineKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)

from bofire.kernels.aggregation import PolynomialFeatureInteractionKernel


@pytest.mark.parametrize(
    "include_self_interactions, diag, last_dim_is_batch",
    list(itertools.product([True, False], repeat=3)),
)
def test_PolynomialFeatureInteractionKernel(
    include_self_interactions, diag, last_dim_is_batch
):
    k1 = RBFKernel()
    k2 = MaternKernel()
    k3 = CosineKernel()

    k_orig = (
        ConstantKernel()
        + ScaleKernel(k1)
        + ScaleKernel(k2)
        + ScaleKernel(k3)
        + ScaleKernel(k1 * k2)
        + ScaleKernel(k1 * k3)
        + ScaleKernel(k2 * k3)
    )
    if include_self_interactions:
        k_orig += ScaleKernel(k1 * k1) + ScaleKernel(k2 * k2) + ScaleKernel(k3 * k3)

    k = PolynomialFeatureInteractionKernel(
        [k1, k2, k3], max_degree=2, include_self_interactions=include_self_interactions
    )

    torch.manual_seed(0)
    x1 = torch.randn(5, 10, 3)
    x2 = torch.randn(5, 10, 3)

    z1 = k_orig(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch).to_dense()
    z2 = k(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch).to_dense()

    assert z1.shape == z2.shape
    assert torch.allclose(z1, z2, atol=1e-6)
