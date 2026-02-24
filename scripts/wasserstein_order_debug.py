import argparse
from dataclasses import dataclass

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.exceptions.errors import OptimizationGradientError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofire.kernels.shape import ExactWassersteinKernel


torch.set_default_dtype(torch.double)


@dataclass
class SegmentExample:
    d0: float
    d1: float
    dx: float


def w1_segment_exact(
    d0: torch.Tensor, d1: torch.Tensor, dx: torch.Tensor
) -> torch.Tensor:
    abs0 = torch.abs(d0)
    abs1 = torch.abs(d1)
    same_sign = (d0 * d1) >= 0
    area_same = 0.5 * (abs0 + abs1) * dx
    area_cross = (
        0.5
        * dx
        * ((abs0 * abs0) + (abs1 * abs1))
        / torch.clamp(
            abs0 + abs1,
            min=1e-12,
        )
    )
    return torch.where(same_sign, area_same, area_cross)


def w2_segment_exact(
    d0: torch.Tensor, d1: torch.Tensor, dx: torch.Tensor
) -> torch.Tensor:
    return torch.sqrt(
        torch.clamp((dx / 3.0) * ((d0 * d0) + (d0 * d1) + (d1 * d1)), min=0.0),
    )


def print_segment_examples() -> None:
    examples = [
        SegmentExample(d0=0.2, d1=0.8, dx=1.0),
        SegmentExample(d0=-0.4, d1=0.6, dx=1.0),
        SegmentExample(d0=-0.7, d1=-0.1, dx=1.0),
    ]
    print("Segment-level W1 vs W2 examples")
    print("d0    d1    dx    W1_segment    W2_segment")
    for ex in examples:
        d0 = torch.tensor(ex.d0)
        d1 = torch.tensor(ex.d1)
        dx = torch.tensor(ex.dx)
        w1 = w1_segment_exact(d0, d1, dx).item()
        w2 = w2_segment_exact(d0, d1, dx).item()
        print(f"{ex.d0: .2f} {ex.d1: .2f} {ex.dx: .2f}   {w1: .6f}      {w2: .6f}")


def build_cdf_inequality_constraints(
    points: int,
    min_dx: float = 1e-3,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Linear constraints for CDF-like candidates.

    Feature layout is [y_0..y_{p-1}, x_0..x_{p-1}].
    """
    constraints: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    # y_{k+1} - y_k >= 0
    for k in range(points - 1):
        idx = torch.tensor([k + 1, k], dtype=torch.long)
        coef = torch.tensor([1.0, -1.0], dtype=torch.double)
        constraints.append((idx, coef, 0.0))

    # x_{k+1} - x_k >= min_dx
    x_offset = points
    for k in range(points - 1):
        idx = torch.tensor([x_offset + k + 1, x_offset + k], dtype=torch.long)
        coef = torch.tensor([1.0, -1.0], dtype=torch.double)
        constraints.append((idx, coef, float(min_dx)))

    return constraints


def target_cdf(x: torch.Tensor) -> torch.Tensor:
    """Nonlinear synthetic CDF-like target."""
    s1 = torch.sigmoid((x - 0.30) / 0.08)
    s2 = torch.sigmoid((x - 0.72) / 0.06)
    y = 0.55 * s1 + 0.45 * s2
    return torch.clamp(y, 0.0, 1.0)


def run_nan_debug(order: int, seed: int, n_init: int, n_iter: int, points: int) -> None:
    torch.manual_seed(seed)

    x_grid = torch.linspace(0.0, 1.0, points)
    x_block = x_grid.unsqueeze(0).repeat(n_init, 1)
    y_init = torch.sort(torch.rand(n_init, points), dim=-1).values
    train_x = torch.cat([y_init, x_block], dim=-1)

    target = target_cdf(x_grid)
    print(f"target cdf on base grid: {['%.6f' % v for v in target.tolist()]}")

    def objective(x: torch.Tensor) -> torch.Tensor:
        y = x[..., :points]
        xvals = x[..., points:]
        y_target = target_cdf(xvals)

        fit_loss = ((y - y_target) ** 2).mean(dim=-1, keepdim=True)
        end_loss = (y[..., :1] ** 2) + ((1.0 - y[..., -1:]) ** 2)
        smooth_loss = torch.abs(y[..., 2:] - 2.0 * y[..., 1:-1] + y[..., :-2]).mean(
            dim=-1,
            keepdim=True,
        )

        total_loss = fit_loss + 0.10 * end_loss + 0.05 * smooth_loss
        return -total_loss

    train_y = objective(train_x)
    eps = 1e-6
    bounds = torch.stack(
        [
            torch.cat([torch.zeros(points * 2)]),
            torch.cat([torch.ones(points * 2)]),
        ]
    )
    inequality_constraints = build_cdf_inequality_constraints(points=points, min_dx=eps)

    print(f"\nNaN debug for order={order}")
    print(f"init best: {train_y.max().item():.6f}")

    for i in range(n_iter):
        kernel = ExactWassersteinKernel(
            idx_x=torch.arange(points, 2 * points, dtype=torch.long),
            idx_y=torch.arange(points, dtype=torch.long),
            prepend_x=torch.tensor([], dtype=torch.double),
            prepend_y=torch.tensor([], dtype=torch.double),
            append_x=torch.tensor([], dtype=torch.double),
            append_y=torch.tensor([], dtype=torch.double),
            normalize_y=torch.tensor([1.0], dtype=torch.double),
            normalize_x=True,
            pair_chunk_size=None,
            order=order,
            ard_num_dims=2 * points,
        )
        model = SingleTaskGP(train_x, train_y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acqf = qLogExpectedImprovement(model=model, best_f=train_y.max())
        try:
            cand, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=8,
                raw_samples=64,
                inequality_constraints=inequality_constraints,
                options={"maxiter": 40, "batch_limit": 4},
            )
            new_y = objective(cand)
            train_x = torch.cat([train_x, cand], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)
            cand_y = cand[0, :points].detach().cpu().tolist()
            cand_x = cand[0, points:].detach().cpu().tolist()
            print(f"iter={i + 1}: success, best={train_y.max().item():.6f}")
            print(f"  y={['%.6f' % v for v in cand_y]}")
            print(f"  x={['%.6f' % v for v in cand_x]}")
        except (OptimizationGradientError, RuntimeError, ValueError) as exc:
            print(f"iter={i + 1}: FAIL ({type(exc).__name__})")
            print(str(exc))
            y = torch.sort(torch.rand(1, points, dtype=torch.double), dim=-1).values
            x = x_grid.unsqueeze(0)
            cand = torch.cat([y, x], dim=-1)
            new_y = objective(cand)
            train_x = torch.cat([train_x, cand], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)
            cand_y = cand[0, :points].detach().cpu().tolist()
            cand_x = cand[0, points:].detach().cpu().tolist()
            print(
                f"iter={i + 1}: fallback random monotone point, best={train_y.max().item():.6f}"
            )
            print(f"  y={['%.6f' % v for v in cand_y]}")
            print(f"  x={['%.6f' % v for v in cand_x]}")
            return

    print("completed all iterations without optimization failure")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug W1/W2 behavior and NaNs in ExactWasserstein BO setup.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--n-iter", type=int, default=6)
    p.add_argument("--points", type=int, default=5)
    p.add_argument(
        "--order",
        type=int,
        choices=[1, 2],
        default=None,
        help="If set, run only this order. Otherwise runs both 1 and 2.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print_segment_examples()

    orders = [args.order] if args.order is not None else [1, 2]
    for order in orders:
        run_nan_debug(
            order=order,
            seed=args.seed,
            n_init=args.n_init,
            n_iter=args.n_iter,
            points=args.points,
        )


if __name__ == "__main__":
    main()
