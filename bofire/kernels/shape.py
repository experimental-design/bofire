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
    # print(
    #     f"[wasserstein] x1 shape={tuple(x1.shape)}, y1 shape={tuple(y1.shape)}, "
    #     f"x2 shape={tuple(x2.shape)}, y2 shape={tuple(y2.shape)}"
    # )
    if x1.dim() != 2 or x2.dim() != 2:
        raise ValueError("Expected x1/x2 to be 2D tensors with shape (n, m).")
    if y1.dim() != 2 or y2.dim() != 2:
        raise ValueError("Expected y1/y2 to be 2D tensors with shape (n, m).")

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    # print(f"[wasserstein] computing pairwise distances: n1={n1}, n2={n2}")
    dists = torch.empty((n1, n2), dtype=x1.dtype, device=x1.device)

    for i in range(n1):
        x1_u, y1_u = x1[i], y1[i]
        for j in range(n2):
            x2_u, y2_u = x2[j], y2[j]
            union_x = torch.sort(torch.cat([x1_u, x2_u])).values
            # if i == 0 and j == 0:
            #     print(
            #         f"[wasserstein] union_x size={union_x.numel()}, "
            #         f"range=({union_x.min().item():.6g}, {union_x.max().item():.6g})"
            #     )
            if union_x.numel() < 2:
                dists[i, j] = torch.zeros((), dtype=x1.dtype, device=x1.device)
                continue
            y1_i = interp1d(x1_u, y1_u, union_x)
            y2_i = interp1d(x2_u, y2_u, union_x)
            diff = torch.abs(y1_i - y2_i)
            # print('[wasserstein] diff shape:', diff)
            # if i == 0 and j == 0:
            #     print(
            #         f"[wasserstein] diff stats: min={diff.min().item():.6g}, "
            #         f"max={diff.max().item():.6g}"
            #     )
            # print an example y1_i and y2_i
            if i == 0 and j == 0:
                print(
                    f"[wasserstein] y1_i shape={tuple(y1_i.shape)}, "
                    f"y2_i shape={tuple(y2_i.shape)}, "
                    f"y1_i {y1_i} "
                    f"y2_i {y2_i}"
                    f"union_x {union_x}"
                )
            dists[i, j] = torch.trapezoid(diff, union_x)

            # print(f'[wasserstein] dists[{i}, {j}] = {dists[i, j].item():.6g}')

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
        # to allow for ard now
        # print one example of x1 and x2
        print(
            f"[WassersteinKernel] x1 shape={tuple(x1.shape)}, x2 shape={tuple(x2.shape)}"
        )
        # print one example of x1 and x2 for instance the first row of the last dimension
        print(f"[WassersteinKernel] x1[0] {x1[0]}, x2[0] {x2[0]}")

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
        print(
            f"[ExactWassersteinKernel] x1 shape={tuple(x1.shape)}, x2 shape={tuple(x2.shape)}, "
            f"diag={diag}, last_dim_is_batch={last_dim_is_batch}"
        )
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
            batch_dists = []
            for b in range(x1.shape[0]):
                x1_x, x1_y = _prepare_piecewise_linear_xy(
                    x1[b],
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
                    x2[b],
                    idx_x,
                    idx_y,
                    prepend_x,
                    prepend_y,
                    append_x,
                    append_y,
                    normalize_y,
                    normalize_x=self.normalize_x,
                )
                # if b == 0:
                #     # print(
                #     #     f"[ExactWassersteinKernel] b=0 x1_x shape={tuple(x1_x.shape)}, "
                #     #     f"x2_x shape={tuple(x2_x.shape)}"
                #     # )
                batch_dists.append(
                    _pairwise_piecewise_linear_wasserstein(x1_x, x1_y, x2_x, x2_y)
                )
            dists = torch.stack(batch_dists, dim=0)
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
            # print(
            #     f"[ExactWassersteinKernel] x1_x shape={tuple(x1_x.shape)}, "
            #     f"x2_x shape={tuple(x2_x.shape)}"
            # )
            dists = _pairwise_piecewise_linear_wasserstein(x1_x, x1_y, x2_x, x2_y)

        print(
            f"[ExactWassersteinKernel] dists shape={tuple(dists.shape)}, "
            f"min={dists.min().item():.6g}, max={dists.max().item():.6g}"
        )

        if self.lengthscale.numel() == 1:
            dists = dists / self.lengthscale.squeeze()
        else:
            dists = dists / self.lengthscale.mean()

        if self.squared:
            return torch.exp(-(dists**2))
        return torch.exp(-dists)
