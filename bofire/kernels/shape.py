from typing import List, Optional, Tuple, Union

import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor

from bofire.utils.torch_tools import interp1d


def _expand_values_like(X: Tensor, values: Tensor) -> Tensor:
    shape = X.shape
    values_reshaped = values.view(*([1] * (len(shape) - 1)), -1)
    return values_reshaped.expand(*shape[:-1], -1).to(X)


def _prepend_values(X: Tensor, values: Tensor) -> Tensor:
    return torch.cat([_expand_values_like(X, values), X], dim=-1)


def _append_values(X: Tensor, values: Tensor) -> Tensor:
    return torch.cat([X, _expand_values_like(X, values)], dim=-1)


def _prepare_piecewise_linear_xy(
    X: Tensor,
    idx_x: Union[List[int], Tensor],
    idx_y: Union[List[int], Tensor],
    prepend_x: Tensor,
    prepend_y: Tensor,
    append_x: Tensor,
    append_y: Tensor,
    normalize_y: Tensor,
    normalize_x: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Extract and normalize x/y for piecewise-linear integration."""
    idx_x_t = torch.as_tensor(idx_x, dtype=torch.long, device=X.device)
    idx_y_t = torch.as_tensor(idx_y, dtype=torch.long, device=X.device)

    x = X[..., idx_x_t]
    y = X[..., idx_y_t]

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


def _pairwise_piecewise_linear_wasserstein(
    x1: Tensor,
    y1: Tensor,
    x2: Tensor,
    y2: Tensor,
) -> Tensor:
    """Pairwise exact integral on a global union grid with sign-aware formula."""
    if x1.dim() != 2 or x2.dim() != 2:
        raise ValueError("Expected x1/x2 to be 2D tensors with shape (n, m).")
    if y1.dim() != 2 or y2.dim() != 2:
        raise ValueError("Expected y1/y2 to be 2D tensors with shape (n, m).")

    union_x = torch.sort(torch.cat([x1.reshape(-1), x2.reshape(-1)])).values

    y1_grid = torch.vmap(interp1d, in_dims=(0, 0, None))(x1, y1, union_x)
    y2_grid = torch.vmap(interp1d, in_dims=(0, 0, None))(x2, y2, union_x)

    dx = union_x[1:] - union_x[:-1]
    d0 = y1_grid[:, None, :-1] - y2_grid[None, :, :-1]
    d1 = y1_grid[:, None, 1:] - y2_grid[None, :, 1:]

    abs0 = torch.abs(d0)
    abs1 = torch.abs(d1)
    same_sign = (d0 * d1) >= 0
    denom = torch.clamp(abs0 + abs1, min=1e-12)

    area_same = 0.5 * (abs0 + abs1) * dx
    area_cross = 0.5 * dx * (abs0**2 + abs1**2) / denom

    area = torch.where(same_sign, area_same, area_cross)
    dists = area.sum(dim=-1)

    return dists


def _pairwise_piecewise_linear_wasserstein_batched(
    x1: Tensor,
    y1: Tensor,
    x2: Tensor,
    y2: Tensor,
) -> Tensor:
    """Batch-wise exact integral on a single global union grid across B and N."""
    if x1.dim() != 3 or x2.dim() != 3:
        raise ValueError("Expected x1/x2 to be 3D tensors with shape (b, n, m).")
    if y1.dim() != 3 or y2.dim() != 3:
        raise ValueError("Expected y1/y2 to be 3D tensors with shape (b, n, m).")
    if x1.shape[0] != x2.shape[0]:
        raise ValueError("Batch dimensions of x1 and x2 must match.")

    bsz, n1, _ = x1.shape
    _, n2, _ = x2.shape

    union_x = torch.sort(torch.cat([x1.reshape(-1), x2.reshape(-1)])).values
    if union_x.numel() < 2:
        return torch.zeros((bsz, n1, n2), dtype=x1.dtype, device=x1.device)

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
    d0 = y1_grid[:, :, None, :-1] - y2_grid[:, None, :, :-1]
    d1 = y1_grid[:, :, None, 1:] - y2_grid[:, None, :, 1:]

    abs0 = torch.abs(d0)
    abs1 = torch.abs(d1)
    same_sign = (d0 * d1) >= 0
    denom = torch.clamp(abs0 + abs1, min=1e-12)

    dx_view = dx.view(1, 1, 1, -1)
    area_same = 0.5 * (abs0 + abs1) * dx_view
    area_cross = 0.5 * dx_view * (abs0**2 + abs1**2) / denom

    area = torch.where(same_sign, area_same, area_cross)
    dists = area.sum(dim=-1)

    return dists


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

        # print("x1 shape:", x1.shape, "x2 shape:", x2.shape)

        dists = self.calc_wasserstein_distances(x1_scaled, x2_scaled)

        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)


class ExactWassersteinKernel(Kernel):
    """Kernel for exact Wasserstein distance of 1D piecewise-linear functions."""

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
        **kwargs,
    ):
        super(ExactWassersteinKernel, self).__init__(**kwargs)
        self.squared = squared
        self.idx_x = idx_x
        self.idx_y = idx_y
        self.prepend_x = prepend_x
        self.prepend_y = prepend_y
        self.append_x = append_x
        self.append_y = append_y
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x

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

        idx_x = self.idx_x.to(device=x1.device)
        idx_y = self.idx_y.to(device=x1.device)
        prepend_x = self.prepend_x.to(x1)
        prepend_y = self.prepend_y.to(x1)
        append_x = self.append_x.to(x1)
        append_y = self.append_y.to(x1)
        normalize_y = self.normalize_y.to(x1)

        if x1.dim() == 3 and x2.dim() == 3:
            if x1.shape[0] != x2.shape[0]:
                raise ValueError("Batch dimensions of x1 and x2 must match.")
            x1_x, x1_y = _prepare_piecewise_linear_xy(
                x1,
                idx_x,
                idx_y,
                prepend_x,
                prepend_y,
                append_x,
                append_y,
                normalize_y,
                normalize_x=self.normalize_x,
            )
            x2_x, x2_y = _prepare_piecewise_linear_xy(
                x2,
                idx_x,
                idx_y,
                prepend_x,
                prepend_y,
                append_x,
                append_y,
                normalize_y,
                normalize_x=self.normalize_x,
            )
            dists = _pairwise_piecewise_linear_wasserstein_batched(
                x1_x, x1_y, x2_x, x2_y
            )
        else:
            x1_x, x1_y = _prepare_piecewise_linear_xy(
                x1,
                idx_x,
                idx_y,
                prepend_x,
                prepend_y,
                append_x,
                append_y,
                normalize_y,
                normalize_x=self.normalize_x,
            )
            x2_x, x2_y = _prepare_piecewise_linear_xy(
                x2,
                idx_x,
                idx_y,
                prepend_x,
                prepend_y,
                append_x,
                append_y,
                normalize_y,
                normalize_x=self.normalize_x,
            )
            dists = _pairwise_piecewise_linear_wasserstein(x1_x, x1_y, x2_x, x2_y)

        if self.lengthscale.numel() == 1:
            dists = dists / self.lengthscale.squeeze()
        else:
            dists = dists / self.lengthscale.mean()

        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)
