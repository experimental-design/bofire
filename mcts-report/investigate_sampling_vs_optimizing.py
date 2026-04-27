"""Investigate the gap between optimized acqf and polytope-sampled acqf values.

Uses FormulationWrapper(benchmark=Hartmann(), max_count=4) which creates a
7D simplex-constrained problem (6 original features + 1 filler, sum-to-1,
NChooseK on the 6 non-filler features).

For each NChooseK subset:
  1. optimize_acqf with linear constraints + fixed features (expensive)
  2. sample_q_batches_from_polytope + evaluate acqf (cheap, hit-and-run)

Questions:
  - How large is the gap between best polytope sample and optimized value?
  - Is there rank correlation between sample-best and optimized rankings?
  - Can cheap samples reliably identify the top subsets?

Usage:
    python mcts-report/investigate_sampling_vs_optimizing.py
"""

import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from botorch.optim import optimize_acqf
from botorch.optim.initializers import sample_q_batches_from_polytope
from botorch.optim.parameter_constraints import _generate_unfixed_lin_constraints
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bofire.data_models.strategies.api as data_models
from bofire.benchmarks.api import Hartmann
from bofire.benchmarks.benchmark import FormulationWrapper
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.strategies.predictives.acqf_optimization import BotorchOptimizer
from bofire.strategies.predictives.sobo import SoboStrategy
from bofire.strategies.random import RandomStrategy
from bofire.strategies.utils import get_torch_bounds_from_domain
from bofire.utils.torch_tools import get_linear_constraints, tkwargs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_COUNT = 4
N_INITIAL = 30
N_RESTARTS = 20
RAW_SAMPLES = 2048
N_SEEDS = 5
SAMPLE_COUNTS = [64, 256, 1024, 2048]


def make_strategy_and_acqf(benchmark, seed: int):
    """Create a fitted SoboStrategy and extract acqf + bounds + constraints."""
    domain = benchmark.domain
    random_strategy = RandomStrategy(
        data_model=data_models.RandomStrategy(domain=domain, seed=seed),
    )
    candidates = random_strategy.ask(N_INITIAL)
    experiments = benchmark.f(candidates, return_complete=True)

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

    acqf = strategy._get_acqfs(1)[0]
    bounds = get_torch_bounds_from_domain(domain, strategy.input_preprocessing_specs)

    # Extract linear constraints in BoTorch format
    ineq_constraints = get_linear_constraints(
        domain=domain,
        constraint=LinearInequalityConstraint,
        unit_scaled=False,
    )
    eq_constraints = get_linear_constraints(
        domain=domain,
        constraint=LinearEqualityConstraint,
        unit_scaled=False,
    )

    best_f = experiments["y"].min()
    return strategy, acqf, bounds, ineq_constraints, eq_constraints, best_f


def get_nchoosek_feature_keys(benchmark):
    """Get the non-filler feature keys and their indices."""
    domain = benchmark.domain
    from bofire.data_models.features.api import ContinuousInput

    all_keys = domain.inputs.get_keys(ContinuousInput)
    nchoosek_keys = [k for k in all_keys if not k.startswith("x_filler_")]
    nchoosek_indices = [all_keys.index(k) for k in nchoosek_keys]
    return nchoosek_keys, nchoosek_indices, all_keys


def enumerate_all_subsets(indices: list[int], max_k: int) -> list[frozenset[int]]:
    """Generate all subsets of indices with size 0..max_k."""
    subsets = []
    for k in range(0, max_k + 1):
        for combo in itertools.combinations(indices, k):
            subsets.append(frozenset(combo))
    return subsets


def optimized_acqf_per_subset(
    acqf,
    bounds: torch.Tensor,
    subsets: list[frozenset[int]],
    nchoosek_indices: set[int],
    ineq_constraints,
    eq_constraints,
) -> dict[frozenset[int], float]:
    """Run optimize_acqf for every subset with linear constraints."""
    dim = bounds.shape[1]
    results = {}

    botorch_ineqs = ineq_constraints if len(ineq_constraints) > 0 else None
    botorch_eqs = eq_constraints if len(eq_constraints) > 0 else None

    for subset in subsets:
        # Fix inactive NChooseK features to 0
        fixed = {i: 0.0 for i in nchoosek_indices if i not in subset}

        if len(subset) == 0:
            # All NChooseK features fixed to 0, only filler is free
            candidate = torch.zeros(1, dim, dtype=bounds.dtype)
            # Set filler to satisfy sum=1 constraint
            filler_indices = [i for i in range(dim) if i not in nchoosek_indices]
            for fi in filler_indices:
                candidate[0, fi] = 1.0 / len(filler_indices)
            with torch.no_grad():
                val = acqf(candidate.unsqueeze(0)).item()
            results[subset] = val
            continue

        try:
            candidates, acq_val = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=N_RESTARTS,
                raw_samples=RAW_SAMPLES,
                fixed_features=fixed,
                inequality_constraints=botorch_ineqs,
                equality_constraints=botorch_eqs,
            )
            results[subset] = acq_val.item()
        except Exception as e:
            print(f"    optimize_acqf failed for {sorted(subset)}: {e}")
            results[subset] = float("-inf")

    return results


def polytope_sample_acqf_per_subset(
    acqf,
    bounds: torch.Tensor,
    subsets: list[frozenset[int]],
    nchoosek_indices: set[int],
    all_continuous_keys: list[str],
    ineq_constraints,
    eq_constraints,
    n_samples: int,
    seed: int,
) -> dict[frozenset[int], dict]:
    """Sample from polytope (hit-and-run) and evaluate acqf for every subset.

    Uses sample_q_batches_from_polytope with _generate_unfixed_lin_constraints
    to handle fixed features, same approach as RandomStrategy._sample_from_polytope.
    """
    dim = bounds.shape[1]
    results = {}

    for subset in subsets:
        # Fix inactive NChooseK features to 0
        fixed_features_dict = {i: 0.0 for i in nchoosek_indices if i not in subset}

        if len(subset) == 0:
            # All NChooseK features fixed, only filler free
            candidate = torch.zeros(1, dim, dtype=bounds.dtype)
            filler_indices = [i for i in range(dim) if i not in nchoosek_indices]
            for fi in filler_indices:
                candidate[0, fi] = 1.0 / len(filler_indices)
            with torch.no_grad():
                val = acqf(candidate.unsqueeze(0)).item()
            results[subset] = {
                "best": val,
                "mean": val,
                "std": 0.0,
                "all": np.array([val]),
            }
            continue

        # Build unfixed bounds (remove fixed dimensions)
        free_indices = [i for i in range(dim) if i not in fixed_features_dict]
        free_lower = bounds[0, free_indices]
        free_upper = bounds[1, free_indices]
        free_bounds = torch.stack([free_lower, free_upper]).to(**tkwargs)

        # Generate unfixed constraints using BoTorch's helper
        unfixed_ineqs = _generate_unfixed_lin_constraints(
            constraints=ineq_constraints,
            eq=False,
            fixed_features=fixed_features_dict,
            dimension=dim,
        )
        unfixed_eqs = _generate_unfixed_lin_constraints(
            constraints=eq_constraints,
            eq=True,
            fixed_features=fixed_features_dict,
            dimension=dim,
        )

        try:
            # Sample from polytope using hit-and-run
            samples = sample_q_batches_from_polytope(
                n=1,
                q=n_samples,
                bounds=free_bounds,
                inequality_constraints=unfixed_ineqs
                if len(unfixed_ineqs) > 0
                else None,
                equality_constraints=unfixed_eqs if len(unfixed_eqs) > 0 else None,
                n_burnin=1000,
                n_thinning=32,
                seed=seed,
            ).squeeze(0)  # (n_samples, free_dim)

            # Reconstruct full-dim candidates
            full_candidates = torch.zeros(n_samples, dim, dtype=bounds.dtype)
            for j, fi in enumerate(free_indices):
                full_candidates[:, fi] = samples[:, j]
            for fi, val in fixed_features_dict.items():
                full_candidates[:, fi] = val

            # Evaluate acqf in batches
            batch_size = 256
            vals = []
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch = full_candidates[i : i + batch_size].unsqueeze(
                        1
                    )  # (b, 1, dim)
                    v = acqf(batch)
                    vals.append(v.cpu().numpy())
            all_vals = np.concatenate(vals)

            results[subset] = {
                "best": float(all_vals.max()),
                "mean": float(all_vals.mean()),
                "std": float(all_vals.std()),
                "all": all_vals,
            }
        except Exception as e:
            print(f"    polytope sample failed for {sorted(subset)}: {e}")
            results[subset] = {
                "best": float("-inf"),
                "mean": float("-inf"),
                "std": 0.0,
                "all": np.array([float("-inf")]),
            }

    return results


def rank_subsets(values: dict[frozenset[int], float]) -> list[frozenset[int]]:
    """Return subsets sorted by value descending."""
    return sorted(values.keys(), key=lambda s: values[s], reverse=True)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*InputDataWarning.*")
    warnings.filterwarnings("ignore", message=".*model inputs.*")
    warnings.filterwarnings("ignore", message=".*not unique.*")

    benchmark = FormulationWrapper(benchmark=Hartmann(), max_count=MAX_COUNT)
    nchoosek_keys, nchoosek_indices, all_keys = get_nchoosek_feature_keys(benchmark)
    nchoosek_set = set(nchoosek_indices)

    subsets = enumerate_all_subsets(nchoosek_indices, MAX_COUNT)
    n_subsets = len(subsets)

    print("Sampling vs Optimizing: FormulationWrapper(Hartmann(), max_count=4)")
    print("=" * 70)
    print(
        f"  Domain: {len(all_keys)} features ({len(nchoosek_keys)} NChooseK + fillers)"
    )
    print(f"  Features: {all_keys}")
    print(f"  NChooseK features: {nchoosek_keys} (indices {nchoosek_indices})")
    print(f"  Total subsets: {n_subsets}")
    print(f"  Constraints: sum-to-1 equality + NChooseK(max={MAX_COUNT})")
    print(f"  Initial points: {N_INITIAL}")
    print(f"  optimize_acqf: {N_RESTARTS} restarts, {RAW_SAMPLES} raw samples")
    print(f"  Sample counts: {SAMPLE_COUNTS}")
    print(f"  Seeds: {N_SEEDS}")
    print()

    # Collect results across seeds
    all_rank_correlations = {n: [] for n in SAMPLE_COUNTS}
    all_top1_match = {n: [] for n in SAMPLE_COUNTS}
    all_top3_overlap = {n: [] for n in SAMPLE_COUNTS}
    all_top5_overlap = {n: [] for n in SAMPLE_COUNTS}
    all_mean_gaps = {n: [] for n in SAMPLE_COUNTS}
    all_winner_gaps = {n: [] for n in SAMPLE_COUNTS}
    all_opt_times = []
    all_sample_times = {n: [] for n in SAMPLE_COUNTS}

    for seed in range(N_SEEDS):
        print(f"{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        strategy, acqf, bounds, ineq_constraints, eq_constraints, best_f = (
            make_strategy_and_acqf(benchmark, seed)
        )
        print(f"  GP fitted, best_f = {best_f:.4f}, bounds shape = {bounds.shape}")

        # 1. Optimized values (gold standard)
        t0 = time.time()
        opt_values = optimized_acqf_per_subset(
            acqf, bounds, subsets, nchoosek_set, ineq_constraints, eq_constraints
        )
        opt_time = time.time() - t0
        all_opt_times.append(opt_time)
        opt_ranking = rank_subsets(opt_values)

        opt_best_subset = opt_ranking[0]
        opt_best_val = opt_values[opt_best_subset]
        print(f"  Exhaustive optimization: {opt_time:.1f}s")
        print(
            f"  Best optimized: subset={sorted(opt_best_subset)} acq={opt_best_val:.4f}"
        )
        print("  Top 5 optimized:")
        for i in range(min(5, n_subsets)):
            s = opt_ranking[i]
            print(f"    #{i+1}: {str(sorted(s)):>20s}  acq = {opt_values[s]:.4f}")

        # 2. Polytope samples at different counts
        for n_samples in SAMPLE_COUNTS:
            t0 = time.time()
            sample_results = polytope_sample_acqf_per_subset(
                acqf,
                bounds,
                subsets,
                nchoosek_set,
                all_keys,
                ineq_constraints,
                eq_constraints,
                n_samples=n_samples,
                seed=seed + 1000,
            )
            sample_time = time.time() - t0
            all_sample_times[n_samples].append(sample_time)

            sample_best_values = {s: r["best"] for s, r in sample_results.items()}
            sample_ranking = rank_subsets(sample_best_values)

            # Rank correlation (Spearman)
            opt_ranks = [opt_ranking.index(s) for s in subsets]
            sample_ranks = [sample_ranking.index(s) for s in subsets]
            rho, _ = stats.spearmanr(opt_ranks, sample_ranks)
            all_rank_correlations[n_samples].append(rho)

            # Top-1 match
            top1_match = sample_ranking[0] == opt_ranking[0]
            all_top1_match[n_samples].append(top1_match)

            # Top-3 overlap
            opt_top3 = set(opt_ranking[:3])
            sample_top3 = set(sample_ranking[:3])
            top3_overlap = len(opt_top3 & sample_top3) / 3
            all_top3_overlap[n_samples].append(top3_overlap)

            # Top-5 overlap
            opt_top5 = set(opt_ranking[:5])
            sample_top5 = set(sample_ranking[:5])
            top5_overlap = len(opt_top5 & sample_top5) / 5
            all_top5_overlap[n_samples].append(top5_overlap)

            # Mean gap per subset (opt - sample_best)
            gaps = [
                opt_values[s] - sample_best_values[s]
                for s in subsets
                if opt_values[s] > float("-inf")
                and sample_best_values[s] > float("-inf")
            ]
            mean_gap = np.mean(gaps) if gaps else float("nan")
            all_mean_gaps[n_samples].append(mean_gap)

            # Winner gap: best opt value - sample winner's sample value
            sample_winner = sample_ranking[0]
            winner_gap = opt_best_val - sample_best_values[sample_winner]
            all_winner_gaps[n_samples].append(winner_gap)

            print(
                f"\n  Polytope samples n={n_samples} ({sample_time:.1f}s, {opt_time/max(sample_time,0.01):.0f}x faster):"
            )
            print(f"    Rank correlation (Spearman rho): {rho:.3f}")
            print(f"    Top-1 match: {top1_match}")
            print(f"    Top-3 overlap: {top3_overlap:.0%}")
            print(f"    Top-5 overlap: {top5_overlap:.0%}")
            print(f"    Mean gap (opt - sample_best): {mean_gap:.4f}")
            print(
                f"    Sample winner: {sorted(sample_winner)} acq={sample_best_values[sample_winner]:.4f} (opt best: {opt_best_val:.4f})"
            )
            print("    Sample top 5:")
            for i in range(min(5, n_subsets)):
                s = sample_ranking[i]
                opt_rank = opt_ranking.index(s) + 1
                print(
                    f"      #{i+1}: {str(sorted(s)):>20s}  "
                    f"sample_best={sample_best_values[s]:.4f}  "
                    f"optimized={opt_values[s]:.4f}  "
                    f"opt_rank=#{opt_rank}"
                )

        print()

    # ---------------------------------------------------------------------------
    # Summary across seeds
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS SEEDS")
    print("=" * 70)
    print(f"  Mean optimize time: {np.mean(all_opt_times):.1f}s")
    print()

    header = f"{'n_samples':>10s} | {'time':>6s} | {'rho':>6s} | {'top1%':>6s} | {'top3%':>6s} | {'top5%':>6s} | {'mean_gap':>8s} | {'winner_gap':>10s}"
    print(header)
    print("-" * len(header))

    for n_samples in SAMPLE_COUNTS:
        t = np.mean(all_sample_times[n_samples])
        rho = np.mean(all_rank_correlations[n_samples])
        top1 = np.mean(all_top1_match[n_samples])
        top3 = np.mean(all_top3_overlap[n_samples])
        top5 = np.mean(all_top5_overlap[n_samples])
        mean_gap = np.mean(all_mean_gaps[n_samples])
        winner_gap = np.mean(all_winner_gaps[n_samples])

        print(
            f"{n_samples:>10d} | {t:>5.1f}s | {rho:>6.3f} | {top1:>5.0%} | {top3:>5.0%} | {top5:>5.0%} | {mean_gap:>8.4f} | {winner_gap:>10.4f}"
        )

    # Per-subset-size gap distribution (last seed, largest sample count)
    n_last = SAMPLE_COUNTS[-1]
    sample_results_last = polytope_sample_acqf_per_subset(
        acqf,
        bounds,
        subsets,
        nchoosek_set,
        all_keys,
        ineq_constraints,
        eq_constraints,
        n_samples=n_last,
        seed=N_SEEDS - 1 + 1000,
    )
    sample_best_last = {s: r["best"] for s, r in sample_results_last.items()}

    print(f"\n{'='*70}")
    print(f"PER-SUBSET-SIZE GAP DISTRIBUTION (last seed, n_samples={n_last})")
    print(f"{'='*70}")

    gaps_by_size = {}
    for s in subsets:
        k = len(s)
        if opt_values[s] > float("-inf") and sample_best_last[s] > float("-inf"):
            gap = opt_values[s] - sample_best_last[s]
            if k not in gaps_by_size:
                gaps_by_size[k] = []
            gaps_by_size[k].append(gap)

    for k in sorted(gaps_by_size.keys()):
        g = gaps_by_size[k]
        print(
            f"  |subset|={k}: n={len(g):>3d}, "
            f"mean_gap={np.mean(g):.4f}, "
            f"median_gap={np.median(g):.4f}, "
            f"max_gap={np.max(g):.4f}"
        )

    # Correlation scatter (top 20 subsets)
    print(f"\n{'='*70}")
    print(f"TOP 20 SUBSETS: optimized vs sample_best (last seed, n={n_last})")
    print(f"{'='*70}")
    print(f"{'subset':>20s} | {'optimized':>10s} | {'sample_best':>11s} | {'gap':>8s}")
    print("-" * 60)
    for s in opt_ranking[:20]:
        ov = opt_values[s]
        sv = sample_best_last[s]
        print(f"{str(sorted(s)):>20s} | {ov:>10.4f} | {sv:>11.4f} | {ov-sv:>8.4f}")

    # Pearson correlation
    opt_arr = np.array(
        [opt_values[s] for s in subsets if opt_values[s] > float("-inf")]
    )
    sample_arr = np.array(
        [sample_best_last[s] for s in subsets if sample_best_last[s] > float("-inf")]
    )
    if len(opt_arr) == len(sample_arr) and len(opt_arr) > 2:
        pearson_r, _ = stats.pearsonr(opt_arr, sample_arr)
        print(f"\n  Pearson r (optimized vs sample_best@{n_last}): {pearson_r:.4f}")


if __name__ == "__main__":
    main()
