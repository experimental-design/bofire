"""Benchmark: Normal-Inverse-Gamma (NIG) MCTS vs Normal-TS vs UCT.

Compares NIG-based MCTS (Student-t posterior) against the best Normal-TS
and UCT configurations on 6 combinatorial NChooseK benchmark problems.

Configurations tested:
  Reference baselines (3):
    1. Random
    2. UCT (+rpol) — best UCT config
    3. TS + TS(g,a) + comb — best Normal-TS config (current recommended default)

  NIG variants (8):
    1. NIG + uniform — minimal NIG, uniform rollout
    2. NIG + TS(g,a) — NIG + learned rollout
    3. NIG + TS(g,a) + comb — NIG + best cache-hit mode
    4. NIG + TS(g,a) + comb + apv — NIG + combined + adaptive variance
    5. NIG + TS(g,a) + vi + apv — NIG + variance inflation + adaptive variance
    6. NIG + TS(g,a) + pess — NIG + pessimistic only
    7. NIG + uniform + pess + apv — NIG + pessimistic + adaptive variance
    8. NIG + TS(g,a) + comb (a0=2) — higher alpha0 = lighter tails

Usage:
    python mcts-report/benchmark_nig.py
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import (
    Problem,
    ProblemResult,
    make_problem_graduated,
    make_problem_large_sparse,
    make_problem_mixed,
    make_problem_multigroup_interaction,
    make_problem_needle,
    make_problem_simple_additive,
    run_random_baseline,
)
from benchmark_ts import TSConfig, UCTConfig, run_ts_config, run_uct_config
from optimize_mcts_nig import MCTS_NIG


OUTPUT_DIR = Path(__file__).parent


# ======================================================================
# Configurations
# ======================================================================


@dataclass
class NIGConfig:
    """Configuration for MCTS_NIG benchmarking."""

    name: str
    nig_alpha0: float = 1.0
    ts_prior_var: float = 1.0
    adaptive_prior_var: bool = False
    cache_hit_mode: str = "no_update"
    variance_decay: float = 0.95
    rollout_mode: str = "uniform"
    pw_k0: float = 2.0
    pw_alpha: float = 0.6
    # Softmax fallback params
    rollout_epsilon: float = 0.3
    rollout_tau: float = 1.0
    rollout_novelty_weight: float = 1.0
    normalize_rewards: bool = True
    adaptive_p_stop: bool = True
    p_stop_rollout: float = 0.35
    p_stop_warmup: int = 20
    p_stop_temperature: float = 0.25


# UCT reference
UCT_REF = UCTConfig(
    name="UCT (+rpol)",
    c_uct=0.01,
    k_rave=0,
    adaptive_p_stop=True,
    normalize_rewards=True,
    rollout_policy=True,
)

# Normal-TS reference (best config from benchmark_ts)
TS_REF = TSConfig(
    name="TS + TS(g,a) + comb",
    rollout_mode="ts_group_action",
    cache_hit_mode="combined",
)

# NIG configs
NIG_CONFIGS = [
    NIGConfig(
        name="NIG + uniform",
        rollout_mode="uniform",
    ),
    NIGConfig(
        name="NIG + TS(g,a)",
        rollout_mode="ts_group_action",
    ),
    NIGConfig(
        name="NIG + TS(g,a) + comb",
        rollout_mode="ts_group_action",
        cache_hit_mode="combined",
    ),
    NIGConfig(
        name="NIG + TS(g,a) + comb + apv",
        rollout_mode="ts_group_action",
        cache_hit_mode="combined",
        adaptive_prior_var=True,
    ),
    NIGConfig(
        name="NIG + TS(g,a) + vi + apv",
        rollout_mode="ts_group_action",
        cache_hit_mode="variance_inflation",
        adaptive_prior_var=True,
    ),
    NIGConfig(
        name="NIG + TS(g,a) + pess",
        rollout_mode="ts_group_action",
        cache_hit_mode="pessimistic",
    ),
    NIGConfig(
        name="NIG + uniform + pess + apv",
        rollout_mode="uniform",
        cache_hit_mode="pessimistic",
        adaptive_prior_var=True,
    ),
    NIGConfig(
        name="NIG + TS(g,a) + comb (a0=2)",
        nig_alpha0=2.0,
        rollout_mode="ts_group_action",
        cache_hit_mode="combined",
    ),
]

ALL_CONFIG_NAMES = ["Random", UCT_REF.name, TS_REF.name] + [c.name for c in NIG_CONFIGS]


# ======================================================================
# Run functions
# ======================================================================


def run_nig_config(problem: Problem, config: NIGConfig, seed: int) -> ProblemResult:
    """Run NIG MCTS with given config, tracking best value per iteration."""
    mcts = MCTS_NIG(
        groups=problem.groups,
        reward_fn=problem.reward_fn,
        nig_alpha0=config.nig_alpha0,
        ts_prior_var=config.ts_prior_var,
        adaptive_prior_var=config.adaptive_prior_var,
        cache_hit_mode=config.cache_hit_mode,
        variance_decay=config.variance_decay,
        rollout_mode=config.rollout_mode,
        pw_k0=config.pw_k0,
        pw_alpha=config.pw_alpha,
        seed=seed,
        rollout_epsilon=config.rollout_epsilon,
        rollout_tau=config.rollout_tau,
        rollout_novelty_weight=config.rollout_novelty_weight,
        normalize_rewards=config.normalize_rewards,
        adaptive_p_stop=config.adaptive_p_stop,
        p_stop_rollout=config.p_stop_rollout,
        p_stop_warmup=config.p_stop_warmup,
        p_stop_temperature=config.p_stop_temperature,
    )

    best_values = []
    for _ in range(problem.n_iterations):
        mcts.run(n_iterations=1)
        best_values.append(mcts.best_value)

    stats = mcts.cache_stats()
    return ProblemResult(
        best_values_over_time=best_values,
        final_best=mcts.best_value,
        found_optimum=abs(mcts.best_value - problem.optimal_value) < 1e-6,
        n_unique_evals=stats["misses"],
    )


# ======================================================================
# Benchmark runner
# ======================================================================


def run_benchmark(problems: list[Problem]):
    """Run all configurations on all problems, collecting results."""
    all_results = {}

    for prob in problems:
        print(f"\n{'='*70}")
        print(f"Problem: {prob.name}")
        print(f"  {prob.description}")
        print(f"  Search space: ~{prob.search_space_size:,} combinations")
        print(f"  Budget: {prob.n_iterations} iterations x {prob.n_trials} trials")
        print(f"{'='*70}")

        # Random baseline
        key = (prob.name, "Random")
        results = []
        t0 = time.time()
        for trial in range(prob.n_trials):
            r = run_random_baseline(prob, seed=trial)
            results.append(r)
        elapsed = time.time() - t0
        all_results[key] = results
        success_rate = sum(r.found_optimum for r in results) / prob.n_trials
        mean_best = np.mean([r.final_best for r in results])
        print(
            f"  {'Random':30s} | best={mean_best:7.1f} | "
            f"opt_rate={success_rate:.0%} | {elapsed:.1f}s"
        )

        # UCT reference
        key = (prob.name, UCT_REF.name)
        results = []
        t0 = time.time()
        for trial in range(prob.n_trials):
            r = run_uct_config(prob, UCT_REF, seed=trial)
            results.append(r)
        elapsed = time.time() - t0
        all_results[key] = results
        success_rate = sum(r.found_optimum for r in results) / prob.n_trials
        mean_best = np.mean([r.final_best for r in results])
        print(
            f"  {UCT_REF.name:30s} | best={mean_best:7.1f} | "
            f"opt_rate={success_rate:.0%} | {elapsed:.1f}s"
        )

        # Normal-TS reference
        key = (prob.name, TS_REF.name)
        results = []
        t0 = time.time()
        for trial in range(prob.n_trials):
            r = run_ts_config(prob, TS_REF, seed=trial)
            results.append(r)
        elapsed = time.time() - t0
        all_results[key] = results
        success_rate = sum(r.found_optimum for r in results) / prob.n_trials
        mean_best = np.mean([r.final_best for r in results])
        print(
            f"  {TS_REF.name:30s} | best={mean_best:7.1f} | "
            f"opt_rate={success_rate:.0%} | {elapsed:.1f}s"
        )

        # NIG configs
        for cfg in NIG_CONFIGS:
            key = (prob.name, cfg.name)
            results = []
            t0 = time.time()
            for trial in range(prob.n_trials):
                r = run_nig_config(prob, cfg, seed=trial)
                results.append(r)
            elapsed = time.time() - t0
            all_results[key] = results
            success_rate = sum(r.found_optimum for r in results) / prob.n_trials
            mean_best = np.mean([r.final_best for r in results])
            print(
                f"  {cfg.name:30s} | best={mean_best:7.1f} | "
                f"opt_rate={success_rate:.0%} | {elapsed:.1f}s"
            )

    return all_results


# ======================================================================
# Plotting
# ======================================================================

COLOR_MAP = {
    "Random": "#888888",
    "UCT (+rpol)": "#1f77b4",
    "TS + TS(g,a) + comb": "#98df8a",
    "NIG + uniform": "#2ca02c",
    "NIG + TS(g,a)": "#d62728",
    "NIG + TS(g,a) + comb": "#9467bd",
    "NIG + TS(g,a) + comb + apv": "#8c564b",
    "NIG + TS(g,a) + vi + apv": "#e377c2",
    "NIG + TS(g,a) + pess": "#ff9896",
    "NIG + uniform + pess + apv": "#ffbb78",
    "NIG + TS(g,a) + comb (a0=2)": "#17becf",
}


def plot_convergence(
    problem_name: str,
    all_results: dict,
    config_names: list[str],
    n_iterations: int,
    suffix: str = "",
    title_extra: str = "",
):
    """Plot mean convergence curves with shaded +/-1 std region."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cname in config_names:
        key = (problem_name, cname)
        if key not in all_results:
            continue
        results = all_results[key]
        curves = np.array([r.best_values_over_time for r in results])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        x = np.arange(1, n_iterations + 1)
        color = COLOR_MAP.get(cname, None)
        ax.plot(x, mean, label=cname, color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=color)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Reward Found", fontsize=12)
    title = f"Convergence: {problem_name}"
    if title_extra:
        title += f" — {title_extra}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"convergence_nig_{problem_name}"
    if suffix:
        fname += f"_{suffix}"
    path = OUTPUT_DIR / f"{fname}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_convergence_subsets(problem_name, all_results, n_iterations):
    """Plot focused subset comparisons for NIG analysis."""
    subsets = {
        "nig_vs_normal_ts": {
            "configs": [
                "Random",
                "UCT (+rpol)",
                "TS + TS(g,a) + comb",
                "NIG + uniform",
                "NIG + TS(g,a)",
                "NIG + TS(g,a) + comb",
            ],
            "title": "NIG vs Normal-TS vs UCT",
        },
        "nig_cache_modes": {
            "configs": [
                "Random",
                "NIG + TS(g,a)",
                "NIG + TS(g,a) + comb",
                "NIG + TS(g,a) + vi + apv",
                "NIG + TS(g,a) + pess",
                "NIG + uniform + pess + apv",
            ],
            "title": "NIG Cache-Hit Modes",
        },
        "nig_alpha": {
            "configs": [
                "Random",
                "TS + TS(g,a) + comb",
                "NIG + TS(g,a) + comb",
                "NIG + TS(g,a) + comb (a0=2)",
                "NIG + TS(g,a) + comb + apv",
            ],
            "title": "NIG Alpha0 & APV Effect",
        },
    }

    for subset_name, spec in subsets.items():
        plot_convergence(
            problem_name,
            all_results,
            spec["configs"],
            n_iterations,
            suffix=subset_name,
            title_extra=spec["title"],
        )


def plot_summary_bar_chart(
    problems: list[Problem], all_results: dict, config_names: list[str]
):
    """Bar chart: mean final best across all problems for each config."""
    fig, axes = plt.subplots(
        1, len(problems), figsize=(5 * len(problems), 6), sharey=False
    )
    if len(problems) == 1:
        axes = [axes]

    for ax, prob in zip(axes, problems):
        names = []
        means = []
        stds = []
        colors = []
        for cname in config_names:
            key = (prob.name, cname)
            if key not in all_results:
                continue
            results = all_results[key]
            finals = [r.final_best for r in results]
            names.append(cname)
            means.append(np.mean(finals))
            stds.append(np.std(finals))
            colors.append(COLOR_MAP.get(cname, "#333333"))

        ax.barh(
            range(len(names)),
            means,
            xerr=stds,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean Final Best Reward", fontsize=10)
        ax.set_title(prob.name, fontsize=11)
        ax.axvline(
            prob.optimal_value,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Optimum",
        )
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("NIG vs Normal-TS vs UCT: Final Best Reward", fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "summary_bar_chart_nig.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_optimum_rate_heatmap(
    problems: list[Problem], all_results: dict, config_names: list[str]
):
    """Heatmap: optimum-finding rate (config x problem)."""
    matrix = []
    for cname in config_names:
        row = []
        for prob in problems:
            key = (prob.name, cname)
            if key not in all_results:
                row.append(0.0)
                continue
            results = all_results[key]
            rate = sum(r.found_optimum for r in results) / len(results)
            row.append(rate)
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(
        figsize=(max(8, len(problems) * 2), max(6, len(config_names) * 0.5))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels([p.name for p in problems], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(config_names)))
    ax.set_yticklabels(config_names, fontsize=9)

    for i in range(len(config_names)):
        for j in range(len(problems)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(
                j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color
            )

    ax.set_title("NIG vs Normal-TS vs UCT: Optimum-Finding Rate", fontsize=13)
    fig.colorbar(im, ax=ax, label="Rate", shrink=0.8)
    fig.tight_layout()
    path = OUTPUT_DIR / "optimum_rate_heatmap_nig.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_unique_evals(
    problems: list[Problem], all_results: dict, config_names: list[str]
):
    """Bar chart: mean number of unique evaluations per config per problem."""
    fig, axes = plt.subplots(
        1, len(problems), figsize=(5 * len(problems), 6), sharey=False
    )
    if len(problems) == 1:
        axes = [axes]

    for ax, prob in zip(axes, problems):
        names = []
        means = []
        colors = []
        for cname in config_names:
            key = (prob.name, cname)
            if key not in all_results:
                continue
            results = all_results[key]
            evals = [r.n_unique_evals for r in results]
            names.append(cname)
            means.append(np.mean(evals))
            colors.append(COLOR_MAP.get(cname, "#333333"))

        ax.barh(
            range(len(names)),
            means,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean Unique Evaluations", fontsize=10)
        ax.set_title(prob.name, fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "NIG vs Normal-TS vs UCT: Unique Selections Evaluated", fontsize=14, y=1.02
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "unique_evals_nig.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_summary_json(problems, all_results, config_names):
    """Save numeric results as JSON for reproducibility."""
    summary = {}
    for prob in problems:
        prob_summary = {}
        for cname in config_names:
            key = (prob.name, cname)
            if key not in all_results:
                continue
            results = all_results[key]
            finals = [r.final_best for r in results]
            opt_rates = [r.found_optimum for r in results]
            unique_evals = [r.n_unique_evals for r in results]
            prob_summary[cname] = {
                "mean_best": float(np.mean(finals)),
                "std_best": float(np.std(finals)),
                "median_best": float(np.median(finals)),
                "optimum_rate": float(np.mean(opt_rates)),
                "mean_unique_evals": float(np.mean(unique_evals)),
            }
        summary[prob.name] = prob_summary

    path = OUTPUT_DIR / "results_nig.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")
    return summary


# ======================================================================
# Main
# ======================================================================


def main():
    print("MCTS Normal-Inverse-Gamma (NIG) Benchmark")
    print("=" * 70)

    problems = [
        make_problem_multigroup_interaction(),
        make_problem_needle(),
        make_problem_mixed(),
        make_problem_large_sparse(),
        make_problem_graduated(),
        make_problem_simple_additive(),
    ]

    for p in problems:
        print(
            f"  {p.name}: ~{p.search_space_size:,} combinations, "
            f"{p.n_iterations} iters x {p.n_trials} trials"
        )

    t_start = time.time()
    all_results = run_benchmark(problems)
    total_time = time.time() - t_start
    print(f"\nTotal benchmark time: {total_time:.1f}s")

    # Generate plots
    print("\nGenerating plots...")
    for prob in problems:
        plot_convergence(prob.name, all_results, ALL_CONFIG_NAMES, prob.n_iterations)
        plot_convergence_subsets(prob.name, all_results, prob.n_iterations)

    plot_summary_bar_chart(problems, all_results, ALL_CONFIG_NAMES)
    plot_optimum_rate_heatmap(problems, all_results, ALL_CONFIG_NAMES)
    plot_unique_evals(problems, all_results, ALL_CONFIG_NAMES)

    # Save numeric results
    summary = save_summary_json(problems, all_results, ALL_CONFIG_NAMES)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    for prob in problems:
        print(
            f"\n{prob.name} (search space: ~{prob.search_space_size:,}, "
            f"optimum: {prob.optimal_value})"
        )
        print(
            f"  {'Config':<35s} {'Mean Best':>10s} {'+-Std':>8s} "
            f"{'Opt Rate':>10s} {'Uniq Evals':>12s}"
        )
        print(f"  {'-'*75}")
        for cname in ALL_CONFIG_NAMES:
            d = summary[prob.name].get(cname)
            if d is None:
                continue
            print(
                f"  {cname:<35s} {d['mean_best']:10.1f} {d['std_best']:8.1f} "
                f"{d['optimum_rate']:10.0%} {d['mean_unique_evals']:12.0f}"
            )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
