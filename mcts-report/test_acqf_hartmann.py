"""Test MCTS acquisition optimization on Hartmann(dim=6, allowed_k=4).

Compares MCTS-guided optimization against exhaustive enumeration of all
NChooseK subsets to verify that MCTS finds the best (or near-best)
combinatorial structure when optimizing a real acquisition function.

Uses bofire's SingleTaskGPSurrogate and SoboStrategy for proper GP fitting
with data transforms, and generates NChooseK-respecting initial data.

Usage:
    python mcts-report/test_acqf_hartmann.py
"""

import itertools
import sys
import time
import warnings
from pathlib import Path

import torch
from botorch.optim import optimize_acqf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bofire.data_models.strategies.api as data_models
from bofire.benchmarks.api import Hartmann
from bofire.data_models.strategies.predictives.acqf_optimization import BotorchOptimizer
from bofire.strategies.predictives.optimize_mcts import optimize_acqf_mcts
from bofire.strategies.predictives.sobo import SoboStrategy
from bofire.strategies.random import RandomStrategy


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DIM = 6
ALLOWED_K = 4
N_INITIAL = 20
N_MCTS_ITERATIONS = 20
N_RESTARTS = 20
RAW_SAMPLES = 2048
N_SEEDS = 5


def make_strategy_and_acqf(benchmark: Hartmann, seed: int):
    """Create a fitted SoboStrategy and extract acqf + bounds.

    Generates NChooseK-respecting initial data via RandomStrategy,
    evaluates on Hartmann6, fits a GP via bofire's SingleTaskGPSurrogate,
    and returns the acquisition function and bounds.
    """
    domain = benchmark.domain

    # Generate NChooseK-respecting initial data
    random_strategy = RandomStrategy(
        data_model=data_models.RandomStrategy(domain=domain, seed=seed),
    )
    candidates = random_strategy.ask(N_INITIAL)
    experiments = benchmark.f(candidates, return_complete=True)

    # Create SoboStrategy with custom optimizer settings
    strategy = SoboStrategy(
        data_model=data_models.SoboStrategy(
            domain=domain,
            acquisition_optimizer=BotorchOptimizer(
                n_restarts=N_RESTARTS,
                n_raw_samples=RAW_SAMPLES,
            ),
        ),
    )
    strategy.tell(experiments)

    # Extract the fitted acqf and bounds
    acqf = strategy._get_acqfs(1)[0]
    # Get bounds in the same way the optimizer does
    from bofire.strategies.utils import get_torch_bounds_from_domain

    bounds = get_torch_bounds_from_domain(domain, strategy.input_preprocessing_specs)

    best_f = experiments["y"].min()  # Hartmann is minimized
    return strategy, acqf, bounds, best_f, experiments


# ---------------------------------------------------------------------------
# Exhaustive enumeration
# ---------------------------------------------------------------------------


def enumerate_all_subsets(dim: int, max_k: int) -> list[frozenset[int]]:
    """Generate all subsets of {0, ..., dim-1} with size 0..max_k."""
    subsets = []
    for k in range(0, max_k + 1):
        for combo in itertools.combinations(range(dim), k):
            subsets.append(frozenset(combo))
    return subsets


def exhaustive_optimize(acqf, bounds: torch.Tensor, subsets: list[frozenset[int]]):
    """Run optimize_acqf for every subset, return best result.

    Returns:
        (best_candidates, best_acq_val, best_subset, all_results)
        where all_results is a list of (subset, acq_val) sorted descending.
    """
    dim = bounds.shape[1]
    results = []

    for subset in subsets:
        # Fix inactive features to 0
        fixed = {i: 0.0 for i in range(dim) if i not in subset}

        if len(subset) == 0:
            # All features fixed to 0 — evaluate directly
            candidate = torch.zeros(1, dim, dtype=bounds.dtype)
            with torch.no_grad():
                val = acqf(candidate.unsqueeze(0)).item()
            results.append((subset, val, candidate))
            continue

        candidates, acq_val = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=N_RESTARTS,
            raw_samples=RAW_SAMPLES,
            fixed_features=fixed,
        )
        results.append((subset, acq_val.item(), candidates))

    # Sort by acquisition value descending
    results.sort(key=lambda x: x[1], reverse=True)

    best_subset, best_val, best_cand = results[0]
    return best_cand, best_val, best_subset, [(s, v) for s, v, _ in results]


# ---------------------------------------------------------------------------
# MCTS optimization
# ---------------------------------------------------------------------------


def mcts_optimize(strategy: SoboStrategy, acqf, bounds: torch.Tensor, seed: int):
    """Run optimize_acqf_mcts and return best result."""
    candidates, acq_val = optimize_acqf_mcts(
        acq_function=acqf,
        bounds=bounds,
        nchooseks=[(list(range(DIM)), 0, ALLOWED_K)],
        num_iterations=N_MCTS_ITERATIONS,
        q=1,
        raw_samples=RAW_SAMPLES,
        num_restarts=N_RESTARTS,
        seed=seed,
    )

    # Determine which features MCTS selected (non-zero in candidates)
    selected = frozenset(i for i in range(DIM) if candidates[0, i].abs().item() > 1e-6)

    return candidates, acq_val, selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*InputDataWarning.*")
    warnings.filterwarnings("ignore", message=".*model inputs.*")

    benchmark = Hartmann(dim=DIM, allowed_k=ALLOWED_K)

    print("MCTS vs Exhaustive Enumeration on Hartmann(dim=6, allowed_k=4)")
    print("=" * 70)
    print(f"  Initial points: {N_INITIAL} (NChooseK-respecting)")
    print(f"  MCTS iterations: {N_MCTS_ITERATIONS}")
    print(f"  Restarts/raw_samples per optimize_acqf: {N_RESTARTS}/{RAW_SAMPLES}")
    print(f"  Seeds: {N_SEEDS}")
    print("  Surrogate: bofire SingleTaskGPSurrogate (with data transforms)")
    print()

    subsets = enumerate_all_subsets(DIM, ALLOWED_K)
    print(f"  Total subsets to enumerate: {len(subsets)}")
    print()

    mcts_ranks = []
    mcts_gaps = []
    mcts_found_best = 0

    for seed in range(N_SEEDS):
        print(f"--- Seed {seed} ---")

        strategy, acqf, bounds, best_f, experiments = make_strategy_and_acqf(
            benchmark, seed
        )
        print(f"  GP trained, best_f = {best_f:.4f}")

        # Exhaustive
        t0 = time.time()
        exh_cand, exh_val, exh_subset, all_results = exhaustive_optimize(
            acqf, bounds, subsets
        )
        exh_time = time.time() - t0

        print(
            f"  Exhaustive: best_acq = {exh_val:.4f}, "
            f"subset = {sorted(exh_subset)}, time = {exh_time:.1f}s"
        )
        print("  Top 5 subsets:")
        for i, (s, v) in enumerate(all_results[:5]):
            print(f"    #{i + 1}: {str(sorted(s)):>20s}  acq = {v:.4f}")

        # MCTS
        t0 = time.time()
        mcts_cand, mcts_val, mcts_subset = mcts_optimize(strategy, acqf, bounds, seed)
        mcts_time = time.time() - t0

        # Find MCTS subset rank in exhaustive results
        rank = next(
            (i + 1 for i, (s, _) in enumerate(all_results) if s == mcts_subset),
            len(all_results),
        )

        gap = exh_val - mcts_val
        mcts_ranks.append(rank)
        mcts_gaps.append(gap)
        if rank == 1:
            mcts_found_best += 1

        print(
            f"  MCTS:       best_acq = {mcts_val:.4f}, "
            f"subset = {sorted(mcts_subset)}, time = {mcts_time:.1f}s"
        )
        print(
            f"  MCTS rank: #{rank}/{len(all_results)}, "
            f"gap = {gap:.4f}, "
            f"speedup = {exh_time / max(mcts_time, 0.01):.1f}x"
        )
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Seeds: {N_SEEDS}")
    print(
        f"  MCTS found best subset: {mcts_found_best}/{N_SEEDS} "
        f"({100 * mcts_found_best / N_SEEDS:.0f}%)"
    )
    print(
        f"  Mean MCTS rank: {sum(mcts_ranks) / len(mcts_ranks):.1f} "
        f"/ {len(subsets)}"
    )
    print(
        f"  Mean acq gap (exhaustive - MCTS): " f"{sum(mcts_gaps) / len(mcts_gaps):.4f}"
    )
    print(f"  MCTS ranks: {mcts_ranks}")


if __name__ == "__main__":
    main()
