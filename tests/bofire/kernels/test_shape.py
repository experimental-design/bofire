import torch
from torch.testing import assert_close

from bofire.kernels.shape import (
    ExactWassersteinKernel,
    WassersteinKernel,
    _build_union_x,
    _pairwise_piecewise_linear_wasserstein,
)
from bofire.utils.torch_tools import interp1d


def _make_monotonic_x(
    batch: int, n: int, points: int, dtype: torch.dtype
) -> torch.Tensor:
    raw = torch.rand(batch, n, points, dtype=dtype)
    return torch.sort(raw, dim=-1).values


def test_pairwise_piecewise_linear_wasserstein_chunked_matches_baseline():
    dtype = torch.double
    batch, n1, n2, points = 2, 3, 4, 8

    x1 = _make_monotonic_x(batch, n1, points, dtype=dtype)
    x2 = _make_monotonic_x(batch, n2, points, dtype=dtype)
    y1 = torch.rand(batch, n1, points, dtype=dtype, requires_grad=True)
    y2 = torch.rand(batch, n2, points, dtype=dtype, requires_grad=True)
    union_x = _build_union_x(x1, x2)

    baseline = _pairwise_piecewise_linear_wasserstein(
        x1,
        y1,
        x2,
        y2,
        union_x=union_x,
        pair_chunk_size=None,
    )
    baseline.sum().backward()
    y1_grad_baseline = y1.grad.detach().clone()
    y2_grad_baseline = y2.grad.detach().clone()

    y1.grad.zero_()
    y2.grad.zero_()

    chunked = _pairwise_piecewise_linear_wasserstein(
        x1,
        y1,
        x2,
        y2,
        union_x=union_x,
        pair_chunk_size=2,
    )
    chunked.sum().backward()

    assert_close(chunked, baseline, rtol=1e-8, atol=1e-10)
    assert_close(y1.grad, y1_grad_baseline, rtol=1e-7, atol=1e-9)
    assert_close(y2.grad, y2_grad_baseline, rtol=1e-7, atol=1e-9)


def test_pairwise_piecewise_linear_wasserstein_auto_chunk_matches_explicit():
    dtype = torch.double
    batch, n1, n2, points = 1, 2, 130, 7

    x1 = _make_monotonic_x(batch, n1, points, dtype=dtype)
    x2 = _make_monotonic_x(batch, n2, points, dtype=dtype)
    y1 = torch.rand(batch, n1, points, dtype=dtype, requires_grad=True)
    y2 = torch.rand(batch, n2, points, dtype=dtype, requires_grad=True)
    union_x = _build_union_x(x1, x2)

    auto = _pairwise_piecewise_linear_wasserstein(
        x1,
        y1,
        x2,
        y2,
        union_x=union_x,
        pair_chunk_size=None,
    )
    auto.sum().backward()
    y1_grad_auto = y1.grad.detach().clone()
    y2_grad_auto = y2.grad.detach().clone()

    y1.grad.zero_()
    y2.grad.zero_()

    explicit = _pairwise_piecewise_linear_wasserstein(
        x1,
        y1,
        x2,
        y2,
        union_x=union_x,
        pair_chunk_size=32,
    )
    explicit.sum().backward()

    assert_close(auto, explicit, rtol=1e-8, atol=1e-10)
    assert_close(y1.grad, y1_grad_auto, rtol=1e-7, atol=1e-9)
    assert_close(y2.grad, y2_grad_auto, rtol=1e-7, atol=1e-9)


def test_pairwise_piecewise_linear_wasserstein_order2_matches_manual():
    dtype = torch.double
    x1 = _make_monotonic_x(batch=1, n=1, points=6, dtype=dtype)
    x2 = _make_monotonic_x(batch=1, n=1, points=5, dtype=dtype)
    y1 = torch.rand(1, 1, 6, dtype=dtype)
    y2 = torch.rand(1, 1, 5, dtype=dtype)
    union_x = _build_union_x(x1, x2)

    d = _pairwise_piecewise_linear_wasserstein(
        x1,
        y1,
        x2,
        y2,
        union_x=union_x,
        pair_chunk_size=None,
        order=2,
    )

    y1_u = interp1d(x1[0, 0], y1[0, 0], union_x)
    y2_u = interp1d(x2[0, 0], y2[0, 0], union_x)
    diff = y1_u - y2_u
    d0 = diff[:-1]
    d1 = diff[1:]
    dx = union_x[1:] - union_x[:-1]
    expected = torch.sqrt(
        torch.clamp(((dx / 3.0) * ((d0 * d0) + (d0 * d1) + (d1 * d1))).sum(), min=0.0)
    )

    assert_close(d.squeeze(), expected, rtol=1e-10, atol=1e-12)


def test_exact_wasserstein_kernel_chunked_matches_baseline():
    dtype = torch.double
    batch, n1, n2, points = 2, 3, 4, 7
    feature_dim = 2 * points
    idx_y = torch.arange(points, dtype=torch.long)
    idx_x = torch.arange(points, 2 * points, dtype=torch.long)

    x1_x = _make_monotonic_x(batch, n1, points, dtype=dtype)
    x2_x = _make_monotonic_x(batch, n2, points, dtype=dtype)
    x1_y = torch.rand(batch, n1, points, dtype=dtype)
    x2_y = torch.rand(batch, n2, points, dtype=dtype)

    x1 = torch.cat([x1_y, x1_x], dim=-1).requires_grad_(True)
    x2 = torch.cat([x2_y, x2_x], dim=-1).requires_grad_(True)

    kwargs = {
        "idx_x": idx_x,
        "idx_y": idx_y,
        "prepend_x": torch.tensor([], dtype=dtype),
        "prepend_y": torch.tensor([], dtype=dtype),
        "append_x": torch.tensor([], dtype=dtype),
        "append_y": torch.tensor([], dtype=dtype),
        "normalize_y": torch.tensor([1.0], dtype=dtype),
        "normalize_x": True,
        "ard_num_dims": feature_dim,
    }
    kernel_baseline = ExactWassersteinKernel(**kwargs, pair_chunk_size=None)
    kernel_chunked = ExactWassersteinKernel(**kwargs, pair_chunk_size=2)
    kernel_chunked.raw_lengthscale.data.copy_(kernel_baseline.raw_lengthscale.data)

    out_baseline = kernel_baseline.forward(x1, x2)
    out_baseline.sum().backward()
    x1_grad_baseline = x1.grad.detach().clone()
    x2_grad_baseline = x2.grad.detach().clone()

    x1.grad.zero_()
    x2.grad.zero_()

    out_chunked = kernel_chunked.forward(x1, x2)
    out_chunked.sum().backward()

    assert_close(out_chunked, out_baseline, rtol=1e-8, atol=1e-10)
    assert_close(x1.grad, x1_grad_baseline, rtol=1e-6, atol=1e-8)
    assert_close(x2.grad, x2_grad_baseline, rtol=1e-6, atol=1e-8)


def test_wasserstein_kernel_diag_matches_matrix_diagonal():
    dtype = torch.double
    x1 = torch.rand(6, 9, dtype=dtype)
    x2 = torch.rand(6, 9, dtype=dtype)
    kernel = WassersteinKernel(ard_num_dims=9)

    full = kernel.forward(x1, x2, diag=False)
    diag = kernel.forward(x1, x2, diag=True)

    assert_close(diag, full.diagonal(dim1=-2, dim2=-1), rtol=1e-8, atol=1e-10)


def test_exact_wasserstein_kernel_diag_matches_matrix_diagonal():
    dtype = torch.double
    batch, n, points = 2, 4, 7
    feature_dim = 2 * points
    idx_y = torch.arange(points, dtype=torch.long)
    idx_x = torch.arange(points, 2 * points, dtype=torch.long)

    x1_x = _make_monotonic_x(batch, n, points, dtype=dtype)
    x2_x = _make_monotonic_x(batch, n, points, dtype=dtype)
    x1_y = torch.rand(batch, n, points, dtype=dtype)
    x2_y = torch.rand(batch, n, points, dtype=dtype)
    x1 = torch.cat([x1_y, x1_x], dim=-1)
    x2 = torch.cat([x2_y, x2_x], dim=-1)

    kernel = ExactWassersteinKernel(
        idx_x=idx_x,
        idx_y=idx_y,
        prepend_x=torch.tensor([], dtype=dtype),
        prepend_y=torch.tensor([], dtype=dtype),
        append_x=torch.tensor([], dtype=dtype),
        append_y=torch.tensor([], dtype=dtype),
        normalize_y=torch.tensor([1.0], dtype=dtype),
        normalize_x=True,
        ard_num_dims=feature_dim,
    )

    full = kernel.forward(x1, x2, diag=False)
    diag = kernel.forward(x1, x2, diag=True)

    assert_close(diag, full.diagonal(dim1=-2, dim2=-1), rtol=1e-8, atol=1e-10)


def test_exact_wasserstein_kernel_order2_chunked_matches_baseline():
    dtype = torch.double
    batch, n1, n2, points = 1, 3, 130, 7
    feature_dim = 2 * points
    idx_y = torch.arange(points, dtype=torch.long)
    idx_x = torch.arange(points, 2 * points, dtype=torch.long)

    x1_x = _make_monotonic_x(batch, n1, points, dtype=dtype)
    x2_x = _make_monotonic_x(batch, n2, points, dtype=dtype)
    x1_y = torch.rand(batch, n1, points, dtype=dtype)
    x2_y = torch.rand(batch, n2, points, dtype=dtype)
    x1 = torch.cat([x1_y, x1_x], dim=-1).requires_grad_(True)
    x2 = torch.cat([x2_y, x2_x], dim=-1).requires_grad_(True)

    kwargs = {
        "idx_x": idx_x,
        "idx_y": idx_y,
        "prepend_x": torch.tensor([], dtype=dtype),
        "prepend_y": torch.tensor([], dtype=dtype),
        "append_x": torch.tensor([], dtype=dtype),
        "append_y": torch.tensor([], dtype=dtype),
        "normalize_y": torch.tensor([1.0], dtype=dtype),
        "normalize_x": True,
        "order": 2,
        "ard_num_dims": feature_dim,
    }
    kernel_auto = ExactWassersteinKernel(**kwargs, pair_chunk_size=None)
    kernel_explicit = ExactWassersteinKernel(**kwargs, pair_chunk_size=32)
    kernel_explicit.raw_lengthscale.data.copy_(kernel_auto.raw_lengthscale.data)

    out_auto = kernel_auto.forward(x1, x2)
    out_auto.sum().backward()
    x1_grad_auto = x1.grad.detach().clone()
    x2_grad_auto = x2.grad.detach().clone()

    x1.grad.zero_()
    x2.grad.zero_()

    out_explicit = kernel_explicit.forward(x1, x2)
    out_explicit.sum().backward()

    assert_close(out_auto, out_explicit, rtol=1e-8, atol=1e-10)
    assert_close(x1.grad, x1_grad_auto, rtol=1e-6, atol=1e-8)
    assert_close(x2.grad, x2_grad_auto, rtol=1e-6, atol=1e-8)
