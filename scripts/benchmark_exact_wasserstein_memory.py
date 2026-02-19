#!/usr/bin/env python3
import argparse
import time
from typing import Dict, Optional

import torch

from bofire.kernels.shape import (
    ExactWassersteinKernel,
    _build_union_x,
    _prepare_piecewise_linear_xy,
)
from bofire.utils.torch_tools import interp1d


def _make_monotonic_x(
    batch: int, n: int, points: int, device: torch.device
) -> torch.Tensor:
    raw = torch.rand(batch, n, points, dtype=torch.double, device=device)
    return torch.sort(raw, dim=-1).values


def _estimate_temp_memory_bytes(
    batch: int,
    n1: int,
    n2: int,
    intervals: int,
    dtype: torch.dtype,
    pair_chunk_size: Optional[int],
) -> int:
    float_bytes = torch.tensor(0, dtype=dtype).element_size()
    bool_bytes = torch.tensor(False).element_size()
    n2_eff = n2 if pair_chunk_size is None else min(n2, pair_chunk_size)
    shape_4d = batch * n1 * n2_eff * intervals
    # Approximate temp tensors in area integration path:
    # d0, d1, abs0, abs1, denom, area_same, area_cross, area (float)
    # same_sign (bool)
    return (8 * float_bytes + bool_bytes) * shape_4d


def _make_kernel(points: int, pair_chunk_size: Optional[int]) -> ExactWassersteinKernel:
    idx_y = torch.arange(points, dtype=torch.long)
    idx_x = torch.arange(points, 2 * points, dtype=torch.long)
    return ExactWassersteinKernel(
        idx_x=idx_x,
        idx_y=idx_y,
        prepend_x=torch.tensor([], dtype=torch.double),
        prepend_y=torch.tensor([], dtype=torch.double),
        append_x=torch.tensor([], dtype=torch.double),
        append_y=torch.tensor([], dtype=torch.double),
        normalize_y=torch.tensor([1.0], dtype=torch.double),
        normalize_x=True,
        ard_num_dims=2 * points,
        pair_chunk_size=pair_chunk_size,
    )


def _pairwise_area_from_grids(
    y1_grid: torch.Tensor,
    y2_grid: torch.Tensor,
    dx: torch.Tensor,
    pair_chunk_size: Optional[int],
    use_view_for_dx: bool,
) -> torch.Tensor:
    bsz, n1, _ = y1_grid.shape
    _, n2, _ = y2_grid.shape
    if pair_chunk_size is not None and pair_chunk_size < 1:
        raise ValueError("pair_chunk_size must be None or >= 1.")

    dx_view = dx.view(1, 1, 1, -1) if use_view_for_dx else dx.reshape(1, 1, 1, -1)
    y1_left = y1_grid[:, :, None, :-1]
    y1_right = y1_grid[:, :, None, 1:]

    if pair_chunk_size is None or pair_chunk_size >= n2:
        d0 = y1_left - y2_grid[:, None, :, :-1]
        d1 = y1_right - y2_grid[:, None, :, 1:]

        abs0 = torch.abs(d0)
        abs1 = torch.abs(d1)
        same_sign = (d0 * d1) >= 0
        denom = torch.clamp(abs0 + abs1, min=1e-12)

        area_same = 0.5 * (abs0 + abs1) * dx_view
        area_cross = 0.5 * dx_view * (abs0**2 + abs1**2) / denom
        area = torch.where(same_sign, area_same, area_cross)
        return area.sum(dim=-1)

    dist_chunks = []
    for start in range(0, n2, pair_chunk_size):
        end = min(start + pair_chunk_size, n2)
        y2_chunk = y2_grid[:, start:end, :]
        d0 = y1_left - y2_chunk[:, None, :, :-1]
        d1 = y1_right - y2_chunk[:, None, :, 1:]

        abs0 = torch.abs(d0)
        abs1 = torch.abs(d1)
        same_sign = (d0 * d1) >= 0
        denom = torch.clamp(abs0 + abs1, min=1e-12)

        area_same = 0.5 * (abs0 + abs1) * dx_view
        area_cross = 0.5 * dx_view * (abs0**2 + abs1**2) / denom
        area = torch.where(same_sign, area_same, area_cross)
        dist_chunks.append(area.sum(dim=-1))

    return torch.cat(dist_chunks, dim=-1)


def _run(
    kernel: ExactWassersteinKernel,
    x1: torch.Tensor,
    x2: torch.Tensor,
    iters: int,
) -> Dict[str, float]:
    device = x1.device
    kernel = kernel.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    last_out = None
    last_x1 = None
    last_x2 = None
    for _ in range(iters):
        x1_i = x1.detach().clone().requires_grad_(True)
        x2_i = x2.detach().clone().requires_grad_(True)
        out = kernel.forward(x1_i, x2_i)
        out.sum().backward()
        last_out = out.detach()
        last_x1 = x1_i.grad.detach()
        last_x2 = x2_i.grad.detach()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = (time.perf_counter() - t0) / iters
    stats: Dict[str, float] = {
        "time_s_per_iter": elapsed,
        "out_mean": float(last_out.mean().item()),
        "x1_grad_norm": float(last_x1.norm().item()),
        "x2_grad_norm": float(last_x2.norm().item()),
    }
    if device.type == "cuda":
        stats["cuda_peak_bytes"] = float(torch.cuda.max_memory_allocated(device))
    return stats


def _run_dx_broadcast_core_benchmark(
    x1: torch.Tensor,
    x2: torch.Tensor,
    points: int,
    pair_chunk_size: Optional[int],
    iters: int,
    use_view_for_dx: bool,
) -> Dict[str, float]:
    device = x1.device
    idx_y = torch.arange(points, dtype=torch.long, device=device)
    idx_x = torch.arange(points, 2 * points, dtype=torch.long, device=device)

    x1_x, x1_y = _prepare_piecewise_linear_xy(
        x1,
        idx_x=idx_x,
        idx_y=idx_y,
        prepend_x=torch.tensor([], dtype=torch.double, device=device),
        prepend_y=torch.tensor([], dtype=torch.double, device=device),
        append_x=torch.tensor([], dtype=torch.double, device=device),
        append_y=torch.tensor([], dtype=torch.double, device=device),
        normalize_y=torch.tensor([1.0], dtype=torch.double, device=device),
        normalize_x=True,
    )
    x2_x, x2_y = _prepare_piecewise_linear_xy(
        x2,
        idx_x=idx_x,
        idx_y=idx_y,
        prepend_x=torch.tensor([], dtype=torch.double, device=device),
        prepend_y=torch.tensor([], dtype=torch.double, device=device),
        append_x=torch.tensor([], dtype=torch.double, device=device),
        append_y=torch.tensor([], dtype=torch.double, device=device),
        normalize_y=torch.tensor([1.0], dtype=torch.double, device=device),
        normalize_x=True,
    )
    union_x = _build_union_x(x1_x, x2_x)
    dx = union_x[1:] - union_x[:-1]

    bsz, n1, _ = x1_x.shape
    _, n2, _ = x2_x.shape
    x1_flat = x1_x.reshape(bsz * n1, -1)
    y1_flat = x1_y.reshape(bsz * n1, -1)
    x2_flat = x2_x.reshape(bsz * n2, -1)
    y2_flat = x2_y.reshape(bsz * n2, -1)
    y1_grid_flat = torch.vmap(interp1d, in_dims=(0, 0, None))(x1_flat, y1_flat, union_x)
    y2_grid_flat = torch.vmap(interp1d, in_dims=(0, 0, None))(x2_flat, y2_flat, union_x)
    y1_grid = y1_grid_flat.reshape(bsz, n1, -1)
    y2_grid = y2_grid_flat.reshape(bsz, n2, -1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    out = None
    grad_norm = None
    for _ in range(iters):
        y1_i = y1_grid.detach().clone().requires_grad_(True)
        y2_i = y2_grid.detach().clone().requires_grad_(True)
        dists = _pairwise_area_from_grids(
            y1_i,
            y2_i,
            dx=dx,
            pair_chunk_size=pair_chunk_size,
            use_view_for_dx=use_view_for_dx,
        )
        out = torch.exp(-dists).sum()
        out.backward()
        grad_norm = (y1_i.grad.norm() + y2_i.grad.norm()).item()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    stats: Dict[str, float] = {
        "time_s_per_iter": (time.perf_counter() - t0) / iters,
        "scalar_out": float(out.detach().item()),
        "grad_norm_sum": float(grad_norm),
    }
    if device.type == "cuda":
        stats["cuda_peak_bytes"] = float(torch.cuda.max_memory_allocated(device))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline vs chunked ExactWassersteinKernel forward/backward "
            "memory characteristics."
        )
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n1", type=int, default=24)
    parser.add_argument("--n2", type=int, default=24)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--compare-dx-view-reshape",
        action="store_true",
        help="Also compare area-core runtime using dx.view(...) vs dx.reshape(...).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device)

    x1_x = _make_monotonic_x(args.batch, args.n1, args.points, device=device)
    x2_x = _make_monotonic_x(args.batch, args.n2, args.points, device=device)
    x1_y = torch.rand(
        args.batch, args.n1, args.points, dtype=torch.double, device=device
    )
    x2_y = torch.rand(
        args.batch, args.n2, args.points, dtype=torch.double, device=device
    )
    x1 = torch.cat([x1_y, x1_x], dim=-1)
    x2 = torch.cat([x2_y, x2_x], dim=-1)

    baseline = _make_kernel(args.points, pair_chunk_size=None).to(device)
    chunked = _make_kernel(args.points, pair_chunk_size=args.chunk_size).to(device)
    chunked.raw_lengthscale.data.copy_(baseline.raw_lengthscale.data)

    base_stats = _run(baseline, x1, x2, iters=args.iters)
    chunk_stats = _run(chunked, x1, x2, iters=args.iters)

    x1_check = x1.detach().clone().requires_grad_(True)
    x2_check = x2.detach().clone().requires_grad_(True)
    out_base = baseline.forward(x1_check, x2_check)
    out_base.sum().backward()
    base_x1_grad = x1_check.grad.detach().clone()
    base_x2_grad = x2_check.grad.detach().clone()

    x1_check.grad.zero_()
    x2_check.grad.zero_()
    out_chunk = chunked.forward(x1_check, x2_check)
    out_chunk.sum().backward()

    idx_y = torch.arange(args.points, dtype=torch.long, device=device)
    idx_x = torch.arange(args.points, 2 * args.points, dtype=torch.long, device=device)
    x1_x_prepared, _ = _prepare_piecewise_linear_xy(
        x1.detach(),
        idx_x=idx_x,
        idx_y=idx_y,
        prepend_x=torch.tensor([], dtype=torch.double, device=device),
        prepend_y=torch.tensor([], dtype=torch.double, device=device),
        append_x=torch.tensor([], dtype=torch.double, device=device),
        append_y=torch.tensor([], dtype=torch.double, device=device),
        normalize_y=torch.tensor([1.0], dtype=torch.double, device=device),
        normalize_x=True,
    )
    x2_x_prepared, _ = _prepare_piecewise_linear_xy(
        x2.detach(),
        idx_x=idx_x,
        idx_y=idx_y,
        prepend_x=torch.tensor([], dtype=torch.double, device=device),
        prepend_y=torch.tensor([], dtype=torch.double, device=device),
        append_x=torch.tensor([], dtype=torch.double, device=device),
        append_y=torch.tensor([], dtype=torch.double, device=device),
        normalize_y=torch.tensor([1.0], dtype=torch.double, device=device),
        normalize_x=True,
    )
    union_len = int(_build_union_x(x1_x_prepared, x2_x_prepared).numel())
    intervals = max(union_len - 1, 0)

    est_base = _estimate_temp_memory_bytes(
        args.batch,
        args.n1,
        args.n2,
        intervals,
        torch.double,
        pair_chunk_size=None,
    )
    est_chunk = _estimate_temp_memory_bytes(
        args.batch,
        args.n1,
        args.n2,
        intervals,
        torch.double,
        pair_chunk_size=args.chunk_size,
    )

    print("ExactWassersteinKernel memory/runtime comparison")
    print(
        f"device={device} batch={args.batch} n1={args.n1} n2={args.n2} points={args.points}"
    )
    print(f"union_grid_points={union_len} intervals={intervals}")
    print("")
    print("Baseline (current implementation)")
    for key, value in base_stats.items():
        print(f"  {key}: {value}")
    print("")
    print(f"Chunked (pair_chunk_size={args.chunk_size})")
    for key, value in chunk_stats.items():
        print(f"  {key}: {value}")
    print("")
    print("Estimated temporary integration memory")
    print(f"  baseline_bytes: {est_base}")
    print(f"  chunked_bytes:  {est_chunk}")
    if est_base > 0:
        print(f"  estimated_reduction: {(1.0 - est_chunk / est_base) * 100.0:.2f}%")
    print("")
    print("Correctness checks")
    print(f"  max_abs_output_diff: {(out_base - out_chunk).abs().max().item()}")
    print(
        f"  max_abs_x1_grad_diff: {(base_x1_grad - x1_check.grad).abs().max().item()}"
    )
    print(
        f"  max_abs_x2_grad_diff: {(base_x2_grad - x2_check.grad).abs().max().item()}"
    )

    if args.compare_dx_view_reshape:
        print("")
        print("dx broadcast op micro-benchmark (area-core only)")
        view_stats = _run_dx_broadcast_core_benchmark(
            x1=x1,
            x2=x2,
            points=args.points,
            pair_chunk_size=args.chunk_size,
            iters=args.iters,
            use_view_for_dx=True,
        )
        reshape_stats = _run_dx_broadcast_core_benchmark(
            x1=x1,
            x2=x2,
            points=args.points,
            pair_chunk_size=args.chunk_size,
            iters=args.iters,
            use_view_for_dx=False,
        )
        print("  using view")
        for key, value in view_stats.items():
            print(f"    {key}: {value}")
        print("  using reshape")
        for key, value in reshape_stats.items():
            print(f"    {key}: {value}")
        if view_stats["time_s_per_iter"] > 0.0:
            rel = (
                (reshape_stats["time_s_per_iter"] - view_stats["time_s_per_iter"])
                / view_stats["time_s_per_iter"]
                * 100.0
            )
            print(f"  reshape_vs_view_time_delta_percent: {rel:.2f}")


if __name__ == "__main__":
    main()
