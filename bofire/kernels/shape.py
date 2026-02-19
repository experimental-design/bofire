from typing import Optional, Tuple

import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor

from bofire.utils.torch_tools import interp1d


_AUTO_CHUNK_THRESHOLD = 128
_AUTO_CHUNK_SIZE = 32
_ORDER2_SQRT_EPS = 1e-12


def _expand_values_like(X: Tensor, values: Tensor) -> Tensor:
    shape = X.shape
    values_reshaped = values.view(*([1] * (len(shape) - 1)), -1)
    return values_reshaped.expand(*shape[:-1], -1).to(X)


def _prepend_values(X: Tensor, values: Tensor) -> Tensor:
    return torch.cat([_expand_values_like(X, values), X], dim=-1)


def _append_values(X: Tensor, values: Tensor) -> Tensor:
    return torch.cat([X, _expand_values_like(X, values)], dim=-1)


def _build_union_x(
    x1: Tensor,
    x2: Tensor,
) -> Tensor:
    union_x = torch.cat([x1.reshape(-1), x2.reshape(-1)])
    if union_x.requires_grad:
        union_x = union_x.detach()
    union_x = torch.sort(union_x).values
    union_x = torch.unique_consecutive(union_x)
    return union_x.contiguous()


def _prepare_piecewise_linear_xy(
    X: Tensor,
    idx_x: Tensor,
    idx_y: Tensor,
    prepend_x: Tensor,
    prepend_y: Tensor,
    append_x: Tensor,
    append_y: Tensor,
    normalize_y: Tensor,
    normalize_x: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Extract and normalize x/y for piecewise-linear integration."""
    x = X[..., idx_x]
    y = X[..., idx_y]

    if prepend_x.numel() > 0:
        x = _prepend_values(x, prepend_x)
    if append_x.numel() > 0:
        x = _append_values(x, append_x)

    if normalize_x:
        x_max = x.max(dim=-1, keepdim=True).values
        x = x / torch.clamp(x_max, min=1e-8)

    if prepend_y.numel() > 0:
        y = _prepend_values(y, prepend_y)
    if append_y.numel() > 0:
        y = _append_values(y, append_y)

    if normalize_y.numel() == 1:
        y = y / normalize_y

    return x, y


def _integrate_piecewise_diff(d0: Tensor, d1: Tensor, dx: Tensor, order: int) -> Tensor:
    if order == 1:
        abs0 = torch.abs(d0)
        abs1 = torch.abs(d1)
        same_sign = (d0 * d1) >= 0
        sum_abs = abs0 + abs1
        denom = torch.clamp(sum_abs, min=1e-12)

        area_same = 0.5 * sum_abs * dx
        sq_sum = (abs0 * abs0) + (abs1 * abs1)
        area_cross = 0.5 * dx * sq_sum / denom
        area = torch.where(same_sign, area_same, area_cross)
        return area.sum(dim=-1)
    if order == 2:
        # Exact integral of a squared linear segment: ∫(a + bt)^2 dt.
        area = (dx / 3.0) * ((d0 * d0) + (d0 * d1) + (d1 * d1))
        return torch.sqrt(torch.clamp(area.sum(dim=-1), min=_ORDER2_SQRT_EPS))
    raise ValueError("order must be 1 or 2.")


def _pairwise_piecewise_linear_wasserstein(
    x1: Tensor,
    y1: Tensor,
    x2: Tensor,
    y2: Tensor,
    union_x: Optional[Tensor] = None,
    pair_chunk_size: Optional[int] = None,
    order: int = 1,
    diag: bool = False,
) -> Tensor:
    """Batch-wise exact integral on a single global union grid across B and N."""
    squeeze_batch = False
    if x1.dim() == 2 and x2.dim() == 2:
        x1 = x1.unsqueeze(0)
        y1 = y1.unsqueeze(0)
        x2 = x2.unsqueeze(0)
        y2 = y2.unsqueeze(0)
        squeeze_batch = True
    if x1.dim() != 3 or x2.dim() != 3:
        raise ValueError("Expected x1/x2 to be 2D or 3D tensors.")
    if y1.dim() != 3 or y2.dim() != 3:
        raise ValueError("Expected y1/y2 to be 2D or 3D tensors.")
    if x1.shape[0] != x2.shape[0]:
        raise ValueError("Batch dimensions of x1 and x2 must match.")

    bsz, n1, _ = x1.shape
    _, n2, _ = x2.shape

    # union_x_with_dups = torch.sort(torch.cat([x1.reshape(-1), x2.reshape(-1)])).values

    # print(
    #     "[pairwise_piecewise_linear_wasserstein_batched] x1 shape:",
    #     x1.shape,
    #     "x2 shape:",
    #     x2.shape,
    # )
    # print(
    #     "[pairwise_piecewise_linear_wasserstein_batched] union_x size:", union_x.shape
    # )
    # print(
    #     "[pairwise_piecewise_linear_wasserstein_batched] union_x_with_dups size:",
    #     union_x_with_dups.shape,
    # )

    if union_x.numel() < 2:
        result = torch.zeros((bsz, n1, n2), dtype=x1.dtype, device=x1.device)
        return result.squeeze(0) if squeeze_batch else result

    if pair_chunk_size is not None and pair_chunk_size < 1:
        raise ValueError("pair_chunk_size must be None or >= 1.")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    effective_chunk_size = pair_chunk_size
    if effective_chunk_size is None and n2 >= _AUTO_CHUNK_THRESHOLD:
        effective_chunk_size = min(_AUTO_CHUNK_SIZE, n2)

    x1_flat = x1.reshape(bsz * n1, -1)
    y1_flat = y1.reshape(bsz * n1, -1)
    x2_flat = x2.reshape(bsz * n2, -1)
    y2_flat = y2.reshape(bsz * n2, -1)

    y1_grid_flat = torch.vmap(interp1d, in_dims=(0, 0, None))(
        x1_flat,
        y1_flat,
        union_x,
    )
    y2_grid_flat = torch.vmap(interp1d, in_dims=(0, 0, None))(
        x2_flat,
        y2_flat,
        union_x,
    )

    y1_grid = y1_grid_flat.reshape(bsz, n1, -1)
    y2_grid = y2_grid_flat.reshape(bsz, n2, -1)

    dx = union_x[1:] - union_x[:-1]
    dx_view = dx.view(1, 1, 1, -1)
    y1_left = y1_grid[:, :, None, :-1]
    y1_right = y1_grid[:, :, None, 1:]

    if diag and n1 == n2:
        d0 = y1_grid[..., :-1] - y2_grid[..., :-1]
        d1 = y1_grid[..., 1:] - y2_grid[..., 1:]
        dists = _integrate_piecewise_diff(d0, d1, dx, order=order)
    elif effective_chunk_size is None or effective_chunk_size >= n2:
        d0 = y1_left - y2_grid[:, None, :, :-1]
        d1 = y1_right - y2_grid[:, None, :, 1:]
        dists = _integrate_piecewise_diff(d0, d1, dx_view, order=order)
    else:
        dist_chunks = []
        for start in range(0, n2, effective_chunk_size):
            end = min(start + effective_chunk_size, n2)
            y2_chunk = y2_grid[:, start:end, :]

            d0 = y1_left - y2_chunk[:, None, :, :-1]
            d1 = y1_right - y2_chunk[:, None, :, 1:]
            dist_chunks.append(
                _integrate_piecewise_diff(d0, d1, dx_view, order=order),
            )

        dists = torch.cat(dist_chunks, dim=-1)

    return dists.squeeze(0) if squeeze_batch else dists


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
        x1_scaled = x1 / self.lengthscale
        x2_scaled = x2 / self.lengthscale

        if diag:
            if x1_scaled.shape == x2_scaled.shape:
                dists = (x1_scaled - x2_scaled).abs().mean(dim=-1).clamp_min(1e-15)
            else:
                dists = self.calc_wasserstein_distances(x1_scaled, x2_scaled).diagonal(
                    dim1=-2,
                    dim2=-1,
                )
        else:
            dists = self.calc_wasserstein_distances(x1_scaled, x2_scaled)

        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)


class ExactWassersteinKernel(Kernel):
    """Kernel for exact Wasserstein distance between 1D piecewise-linear functions."""

    has_lengthscale = True

    def __init__(
        self,
        squared: bool = False,
        idx_x: Optional[Tensor] = None,
        idx_y: Optional[Tensor] = None,
        prepend_x: Optional[Tensor] = None,
        prepend_y: Optional[Tensor] = None,
        append_x: Optional[Tensor] = None,
        append_y: Optional[Tensor] = None,
        normalize_y: Optional[Tensor] = None,
        normalize_x: bool = True,
        pair_chunk_size: Optional[int] = None,
        order: int = 1,
        **kwargs,
    ):
        super(ExactWassersteinKernel, self).__init__(**kwargs)
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2.")
        self.squared = squared
        self.register_buffer("idx_x", idx_x)
        self.register_buffer("idx_y", idx_y)
        self.register_buffer("prepend_x", prepend_x)
        self.register_buffer("prepend_y", prepend_y)
        self.register_buffer("append_x", append_x)
        self.register_buffer("append_y", append_y)
        self.register_buffer("normalize_y", normalize_y)
        self.normalize_x = normalize_x
        self.pair_chunk_size = pair_chunk_size
        self.order = order

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        if self.idx_x is None or self.idx_y is None:
            raise RuntimeError("ExactWassersteinKernel is missing x/y index settings.")
        if (
            self.prepend_x is None
            or self.prepend_y is None
            or self.append_x is None
            or self.append_y is None
            or self.normalize_y is None
        ):
            raise RuntimeError(
                "ExactWassersteinKernel is missing prepend/append/normalize settings."
            )

        x1_x, x1_y = _prepare_piecewise_linear_xy(
            x1,
            self.idx_x,
            self.idx_y,
            self.prepend_x,
            self.prepend_y,
            self.append_x,
            self.append_y,
            self.normalize_y,
            normalize_x=self.normalize_x,
        )
        x2_x, x2_y = _prepare_piecewise_linear_xy(
            x2,
            self.idx_x,
            self.idx_y,
            self.prepend_x,
            self.prepend_y,
            self.append_x,
            self.append_y,
            self.normalize_y,
            normalize_x=self.normalize_x,
        )
        union_x = _build_union_x(x1_x, x2_x)

        # print("[ExactWassersteinKernel] union_x size:", union_x.shape, "union_x:", union_x)
        dists = _pairwise_piecewise_linear_wasserstein(
            x1_x,
            x1_y,
            x2_x,
            x2_y,
            union_x=union_x,
            pair_chunk_size=self.pair_chunk_size,
            order=self.order,
            diag=diag and x1_x.shape[-2] == x2_x.shape[-2],
        )
        if diag and x1_x.shape[-2] != x2_x.shape[-2]:
            dists = dists.diagonal(dim1=-2, dim2=-1)

        if self.lengthscale.numel() == 1:
            dists = dists / self.lengthscale.squeeze()
        else:
            dists = dists / self.lengthscale.mean()

        dists = dists.clamp_min(1e-15)

        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)
