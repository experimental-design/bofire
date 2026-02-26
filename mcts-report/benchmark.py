"""MCTS Benchmark: Comparing MCTS configurations on combinatorial NChooseK problems.

Tests RAVE on/off, Progressive Widening on/off, exploration constants,
and p_stop_rollout against a random-sampling baseline across multiple
problem instances with varying combinatorial complexity and reward structure.

Usage:
    python mcts-report/benchmark.py
"""

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bofire.strategies.predictives.optimize_mcts import (
    MCTS,
    Categorical,
    Groups,
    NChooseK,
)


OUTPUT_DIR = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════════════
# Benchmark problems
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ProblemResult:
    """Results for a single trial of a single configuration on one problem."""

    best_values_over_time: list[float]  # best value at each iteration
    final_best: float
    found_optimum: bool
    n_unique_evals: int  # distinct combinatorial selections evaluated


@dataclass
class Problem:
    """A benchmark problem definition."""

    name: str
    description: str
    groups: Groups
    reward_fn: object  # Callable[[tuple[int,...], dict[int,float]], float]
    optimal_value: float
    search_space_size: int  # approximate number of feasible selections
    n_iterations: int  # budget per trial
    n_trials: int  # number of random seeds


def count_nchoosek_combos(n: int, min_k: int, max_k: int) -> int:
    """Count the number of subsets of size min_k..max_k from n items."""
    return sum(math.comb(n, k) for k in range(min_k, max_k + 1))


# ---------------------------------------------------------------------------
# Problem 1: Multi-group feature selection with pairwise interactions
# ---------------------------------------------------------------------------
def make_problem_multigroup_interaction() -> Problem:
    """3 NChooseK groups (8 features each, pick 1-4) with cross-group interactions.

    Search space: (C(8,1)+...+C(8,4))^3 = 162^3 ≈ 4.25M combinations.
    Reward has partial credit per correct feature + interaction bonuses.
    """
    g1 = NChooseK(features=list(range(0, 8)), min_count=1, max_count=4)
    g2 = NChooseK(features=list(range(8, 16)), min_count=1, max_count=4)
    g3 = NChooseK(features=list(range(16, 24)), min_count=1, max_count=4)
    gs = Groups(groups=[g1, g2, g3])

    # Optimal: {1, 5} from g1, {9, 14} from g2, {17, 20, 23} from g3
    opt_g1 = {1, 5}
    opt_g2 = {9, 14}
    opt_g3 = {17, 20, 23}
    optimal_set = opt_g1 | opt_g2 | opt_g3

    def reward_fn(feats, _cats):
        feat_set = set(feats)
        # Base: partial credit per correct feature
        correct = len(feat_set & optimal_set)
        wrong = len(feat_set - optimal_set)
        score = correct * 8.0 - wrong * 3.0

        # Interaction bonuses (cross-group pairs)
        if 1 in feat_set and 9 in feat_set:
            score += 12.0
        if 5 in feat_set and 14 in feat_set:
            score += 12.0
        if 9 in feat_set and 20 in feat_set:
            score += 12.0
        if 14 in feat_set and 17 in feat_set:
            score += 10.0
        if 1 in feat_set and 23 in feat_set:
            score += 10.0

        # Exact-match bonus
        if feat_set == optimal_set:
            score = 150.0

        return score

    ss = count_nchoosek_combos(8, 1, 4) ** 3  # 162^3
    return Problem(
        name="multigroup_interaction",
        description="3 groups × 8 features (pick 1-4), cross-group interactions",
        groups=gs,
        reward_fn=reward_fn,
        optimal_value=150.0,
        search_space_size=ss,
        n_iterations=600,
        n_trials=30,
    )


# ---------------------------------------------------------------------------
# Problem 2: Needle in a haystack — wide single group
# ---------------------------------------------------------------------------
def make_problem_needle() -> Problem:
    """Single NChooseK group: 15 features, pick 2-5.

    Search space: C(15,2)+...+C(15,5) = 4928 combinations.
    Only one specific subset gives high reward; slight partial credit.
    """
    g = NChooseK(features=list(range(15)), min_count=2, max_count=5)
    gs = Groups(groups=[g])

    target = {3, 7, 11}

    def reward_fn(feats, _cats):
        feat_set = set(feats)
        if feat_set == target:
            return 100.0
        overlap = len(feat_set & target)
        extras = len(feat_set - target)
        return overlap * 15.0 - extras * 5.0

    ss = count_nchoosek_combos(15, 2, 5)
    return Problem(
        name="needle_in_haystack",
        description="15 features pick 2-5, single optimal subset",
        groups=gs,
        reward_fn=reward_fn,
        optimal_value=100.0,
        search_space_size=ss,
        n_iterations=400,
        n_trials=30,
    )


# ---------------------------------------------------------------------------
# Problem 3: Mixed NChooseK + Categorical with interactions
# ---------------------------------------------------------------------------
def make_problem_mixed() -> Problem:
    """2 NChooseK groups + 2 Categorical dimensions with interactions.

    NChooseK: 6 features pick 1-3 each (C(6,1)+C(6,2)+C(6,3)=41 per group).
    Categorical: 4 values each (4×4=16 combos).
    Total: ~41×41×16 ≈ 26,896 combinations.
    """
    g1 = NChooseK(features=list(range(0, 6)), min_count=1, max_count=3)
    g2 = NChooseK(features=list(range(6, 12)), min_count=1, max_count=3)
    cat1 = Categorical(dim=20, values=[0.0, 1.0, 2.0, 3.0])
    cat2 = Categorical(dim=21, values=[0.0, 1.0, 2.0, 3.0])
    gs = Groups(groups=[g1, g2, cat1, cat2])

    opt_feats = {2, 4, 8, 11}
    opt_cats = {20: 2.0, 21: 3.0}

    def reward_fn(feats, cats):
        feat_set = set(feats)
        # Feature credit
        correct_feats = len(feat_set & opt_feats)
        wrong_feats = len(feat_set - opt_feats)
        score = correct_feats * 10.0 - wrong_feats * 4.0

        # Categorical credit
        for dim, val in opt_cats.items():
            if cats.get(dim) == val:
                score += 12.0

        # Interaction: feature 2 + cat 20=2.0
        if 2 in feat_set and cats.get(20) == 2.0:
            score += 15.0

        # Interaction: feature 11 + cat 21=3.0
        if 11 in feat_set and cats.get(21) == 3.0:
            score += 15.0

        # Exact match bonus
        if feat_set == opt_feats and cats == opt_cats:
            score = 150.0

        return score

    ss = count_nchoosek_combos(6, 1, 3) ** 2 * 4 * 4
    return Problem(
        name="mixed_nchoosek_categorical",
        description="2 NChooseK (6 feat, 1-3) + 2 Categorical (4 vals each)",
        groups=gs,
        reward_fn=reward_fn,
        optimal_value=150.0,
        search_space_size=ss,
        n_iterations=500,
        n_trials=30,
    )


# ---------------------------------------------------------------------------
# Problem 4: Large-scale sparse selection
# ---------------------------------------------------------------------------
def make_problem_large_sparse() -> Problem:
    """4 NChooseK groups (10 features each, pick 0-3).

    This tests MCTS with min_count=0 (selecting nothing from a group is valid).
    Search space: (C(10,0)+C(10,1)+C(10,2)+C(10,3))^4 = 176^4 ≈ 960M.
    Optimal uses features from only 2 of the 4 groups.
    """
    g1 = NChooseK(features=list(range(0, 10)), min_count=0, max_count=3)
    g2 = NChooseK(features=list(range(10, 20)), min_count=0, max_count=3)
    g3 = NChooseK(features=list(range(20, 30)), min_count=0, max_count=3)
    g4 = NChooseK(features=list(range(30, 40)), min_count=0, max_count=3)
    gs = Groups(groups=[g1, g2, g3, g4])

    # Optimal: select from groups 1 and 3 only
    opt_g1 = {2, 7}
    opt_g3 = {22, 25, 28}
    optimal_set = opt_g1 | opt_g3

    def reward_fn(feats, _cats):
        feat_set = set(feats)
        if feat_set == optimal_set:
            return 200.0
        correct = len(feat_set & optimal_set)
        wrong = len(feat_set - optimal_set)
        score = correct * 12.0 - wrong * 6.0

        # Bonus for sparsity (using fewer groups)
        groups_used = set()
        for f in feat_set:
            groups_used.add(f // 10)
        if len(groups_used) <= 2:
            score += 8.0
        return score

    ss = count_nchoosek_combos(10, 0, 3) ** 4
    return Problem(
        name="large_sparse",
        description="4 groups × 10 features (pick 0-3), optimal uses only 2 groups",
        groups=gs,
        reward_fn=reward_fn,
        optimal_value=200.0,
        search_space_size=ss,
        n_iterations=800,
        n_trials=30,
    )


# ---------------------------------------------------------------------------
# Problem 5: Overlapping reward landscape (many near-optimal solutions)
# ---------------------------------------------------------------------------
def make_problem_graduated() -> Problem:
    """10 features, pick 2-4. Smooth reward landscape based on feature quality scores.

    Each feature has a fixed quality; reward = sum of qualities of selected features
    minus a penalty for extra features. The landscape is smooth so MCTS benefits
    from learning which features are generally good.
    """
    g = NChooseK(features=list(range(10)), min_count=2, max_count=4)
    gs = Groups(groups=[g])

    # Feature quality scores (pre-determined)
    quality = {
        0: 5.0,
        1: 12.0,
        2: 3.0,
        3: 18.0,
        4: 7.0,
        5: 15.0,
        6: 2.0,
        7: 20.0,
        8: 9.0,
        9: 11.0,
    }
    # Optimal: {3, 7, 5} with reward 18+20+15 = 53, or {1,3,5,7} = 12+18+15+20=65
    # Actually {1,3,5,7} = 65 is best 4-subset

    def reward_fn(feats, _cats):
        return sum(quality[f] for f in feats)

    ss = count_nchoosek_combos(10, 2, 4)
    optimal_val = sum(sorted(quality.values(), reverse=True)[:4])  # top 4
    return Problem(
        name="graduated_landscape",
        description="10 features pick 2-4, smooth quality-based reward",
        groups=gs,
        reward_fn=reward_fn,
        optimal_value=optimal_val,
        search_space_size=ss,
        n_iterations=300,
        n_trials=30,
    )


# ═══════════════════════════════════════════════════════════════════════
# MCTS Configurations
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MCTSConfig:
    """A named MCTS configuration for benchmarking."""

    name: str
    c_uct: float = 1.0
    k_rave: float = 300.0
    p_stop_rollout: float = 0.35
    pw_k0: float = 2.0
    pw_alpha: float = 0.6
    adaptive_p_stop: bool = False
    p_stop_warmup: int = 20
    p_stop_temperature: float = 0.25
    normalize_rewards: bool = False
    rollout_policy: bool = False
    rollout_epsilon: float = 0.3
    rollout_tau: float = 1.0
    rollout_novelty_weight: float = 1.0
    context_rave: bool = False


# Effective ways to disable features:
# - RAVE off: k_rave=0 → beta=0 → pure UCT
# - PW off: pw_k0=1e6 → child_limit always exceeds legal actions
CONFIGS = [
    MCTSConfig(name="MCTS (default)", c_uct=1.0, k_rave=300, pw_k0=2.0, pw_alpha=0.6),
    MCTSConfig(name="MCTS (no RAVE)", c_uct=1.0, k_rave=0, pw_k0=2.0, pw_alpha=0.6),
    MCTSConfig(name="MCTS (no PW)", c_uct=1.0, k_rave=300, pw_k0=1e6, pw_alpha=0.6),
    MCTSConfig(
        name="MCTS (no RAVE, no PW)", c_uct=1.0, k_rave=0, pw_k0=1e6, pw_alpha=0.6
    ),
    MCTSConfig(
        name="MCTS (low explore)", c_uct=0.1, k_rave=300, pw_k0=2.0, pw_alpha=0.6
    ),
    MCTSConfig(
        name="MCTS (high explore)", c_uct=5.0, k_rave=300, pw_k0=2.0, pw_alpha=0.6
    ),
    MCTSConfig(
        name="MCTS (heavy RAVE)", c_uct=1.0, k_rave=3000, pw_k0=2.0, pw_alpha=0.6
    ),
    MCTSConfig(name="MCTS (tight PW)", c_uct=1.0, k_rave=300, pw_k0=1.0, pw_alpha=0.4),
    MCTSConfig(name="MCTS (loose PW)", c_uct=1.0, k_rave=300, pw_k0=5.0, pw_alpha=0.8),
    MCTSConfig(
        name="MCTS (p_stop=0.1)",
        c_uct=1.0,
        k_rave=300,
        pw_k0=2.0,
        pw_alpha=0.6,
        p_stop_rollout=0.1,
    ),
    MCTSConfig(
        name="MCTS (p_stop=0.6)",
        c_uct=1.0,
        k_rave=300,
        pw_k0=2.0,
        pw_alpha=0.6,
        p_stop_rollout=0.6,
    ),
    MCTSConfig(
        name="MCTS (adaptive p)",
        c_uct=1.0,
        k_rave=300,
        pw_k0=2.0,
        pw_alpha=0.6,
        adaptive_p_stop=True,
    ),
    MCTSConfig(
        name="MCTS (no RAVE+adpt)",
        c_uct=1.0,
        k_rave=0,
        pw_k0=2.0,
        pw_alpha=0.6,
        adaptive_p_stop=True,
    ),
    MCTSConfig(
        name="MCTS (norm)",
        c_uct=0.01,
        k_rave=300,
        pw_k0=2.0,
        pw_alpha=0.6,
        normalize_rewards=True,
    ),
    MCTSConfig(
        name="MCTS (no RAVE+adpt+norm)",
        c_uct=0.01,
        k_rave=0,
        pw_k0=2.0,
        pw_alpha=0.6,
        adaptive_p_stop=True,
        normalize_rewards=True,
    ),
    MCTSConfig(
        name="MCTS (+rpol)",
        c_uct=0.01,
        k_rave=0,
        adaptive_p_stop=True,
        normalize_rewards=True,
        rollout_policy=True,
    ),
    MCTSConfig(
        name="MCTS (+rpol ε=0.1)",
        c_uct=0.01,
        k_rave=0,
        adaptive_p_stop=True,
        normalize_rewards=True,
        rollout_policy=True,
        rollout_epsilon=0.1,
    ),
    MCTSConfig(
        name="MCTS (+rpol τ=0.5)",
        c_uct=0.01,
        k_rave=0,
        adaptive_p_stop=True,
        normalize_rewards=True,
        rollout_policy=True,
        rollout_tau=0.5,
    ),
    MCTSConfig(
        name="MCTS (+rpol τ=2)",
        c_uct=0.01,
        k_rave=0,
        adaptive_p_stop=True,
        normalize_rewards=True,
        rollout_policy=True,
        rollout_tau=2.0,
    ),
    MCTSConfig(
        name="MCTS (+crave k=100)",
        c_uct=0.01,
        adaptive_p_stop=True,
        normalize_rewards=True,
        rollout_policy=True,
        context_rave=True,
        k_rave=100,
    ),
    MCTSConfig(
        name="MCTS (+crave k=300)",
        c_uct=0.01,
        adaptive_p_stop=True,
        normalize_rewards=True,
        rollout_policy=True,
        context_rave=True,
        k_rave=300,
    ),
    MCTSConfig(
        name="MCTS (+crave k=500)",
        c_uct=0.01,
        adaptive_p_stop=True,
        normalize_rewards=True,
        rollout_policy=True,
        context_rave=True,
        k_rave=500,
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Random baseline
# ═══════════════════════════════════════════════════════════════════════


def run_random_baseline(problem: Problem, seed: int) -> ProblemResult:
    """Random rollouts from root node, tracking best value per iteration."""
    # Create a dummy MCTS just to use its rollout machinery
    mcts_tmp = MCTS(
        groups=problem.groups,
        reward_fn=lambda f, c: 0.0,
        rollout_policy=False,
        seed=seed,
    )
    rng = random.Random(seed)

    best = float("-inf")
    best_values = []
    seen = set()

    for _ in range(problem.n_iterations):
        mcts_tmp.rng = rng
        feats, cats, _traj = mcts_tmp._rollout(mcts_tmp.root)
        val = problem.reward_fn(feats, cats)
        key = (feats, frozenset(cats.items()))
        seen.add(key)
        if val > best:
            best = val
        best_values.append(best)

    return ProblemResult(
        best_values_over_time=best_values,
        final_best=best,
        found_optimum=abs(best - problem.optimal_value) < 1e-6,
        n_unique_evals=len(seen),
    )


# ═══════════════════════════════════════════════════════════════════════
# MCTS run
# ═══════════════════════════════════════════════════════════════════════


def run_mcts_config(problem: Problem, config: MCTSConfig, seed: int) -> ProblemResult:
    """Run MCTS with given config, tracking best value per iteration."""
    mcts = MCTS(
        groups=problem.groups,
        reward_fn=problem.reward_fn,
        c_uct=config.c_uct,
        k_rave=config.k_rave,
        p_stop_rollout=config.p_stop_rollout,
        pw_k0=config.pw_k0,
        pw_alpha=config.pw_alpha,
        adaptive_p_stop=config.adaptive_p_stop,
        p_stop_warmup=config.p_stop_warmup,
        p_stop_temperature=config.p_stop_temperature,
        normalize_rewards=config.normalize_rewards,
        rollout_policy=config.rollout_policy,
        rollout_epsilon=config.rollout_epsilon,
        rollout_tau=config.rollout_tau,
        rollout_novelty_weight=config.rollout_novelty_weight,
        context_rave=config.context_rave,
        seed=seed,
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
        n_unique_evals=stats["misses"],  # cache misses = unique evaluations
    )


# ═══════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════


def run_benchmark(problems: list[Problem], configs: list[MCTSConfig]):
    """Run all configurations on all problems, collecting results."""
    all_results = {}  # (problem_name, config_name) -> list[ProblemResult]

    for prob in problems:
        print(f"\n{'='*70}")
        print(f"Problem: {prob.name}")
        print(f"  {prob.description}")
        print(f"  Search space: ~{prob.search_space_size:,} combinations")
        print(f"  Budget: {prob.n_iterations} iterations × {prob.n_trials} trials")
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
            f"  Random          | best={mean_best:7.1f} | opt_rate={success_rate:.0%} | {elapsed:.1f}s"
        )

        # MCTS configs
        for cfg in configs:
            key = (prob.name, cfg.name)
            results = []
            t0 = time.time()
            for trial in range(prob.n_trials):
                r = run_mcts_config(prob, cfg, seed=trial)
                results.append(r)
            elapsed = time.time() - t0
            all_results[key] = results
            success_rate = sum(r.found_optimum for r in results) / prob.n_trials
            mean_best = np.mean([r.final_best for r in results])
            print(
                f"  {cfg.name:17s} | best={mean_best:7.1f} | opt_rate={success_rate:.0%} | {elapsed:.1f}s"
            )

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

# Consistent color scheme
COLOR_MAP = {
    "Random": "#888888",
    "MCTS (default)": "#1f77b4",
    "MCTS (no RAVE)": "#ff7f0e",
    "MCTS (no PW)": "#2ca02c",
    "MCTS (no RAVE, no PW)": "#d62728",
    "MCTS (low explore)": "#9467bd",
    "MCTS (high explore)": "#8c564b",
    "MCTS (heavy RAVE)": "#e377c2",
    "MCTS (tight PW)": "#7f7f7f",
    "MCTS (loose PW)": "#bcbd22",
    "MCTS (p_stop=0.1)": "#17becf",
    "MCTS (p_stop=0.6)": "#aec7e8",
    "MCTS (adaptive p)": "#ff1493",
    "MCTS (no RAVE+adpt)": "#00ced1",
    "MCTS (norm)": "#ff6347",
    "MCTS (no RAVE+adpt+norm)": "#32cd32",
    "MCTS (+rpol)": "#8b0000",
    "MCTS (+rpol ε=0.1)": "#ff4500",
    "MCTS (+rpol τ=0.5)": "#daa520",
    "MCTS (+rpol τ=2)": "#4682b4",
    "MCTS (+crave k=100)": "#6a0dad",
    "MCTS (+crave k=300)": "#20b2aa",
    "MCTS (+crave k=500)": "#dc143c",
}


def plot_convergence_curves(
    problem_name: str,
    all_results: dict,
    config_names: list[str],
    n_iterations: int,
):
    """Plot mean convergence curves with shaded ±1 std region."""
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
    ax.set_title(f"Convergence: {problem_name}", fontsize=14)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / f"convergence_{problem_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_convergence_subsets(problem_name, all_results, n_iterations):
    """Plot focused convergence comparisons: RAVE effect, PW effect, exploration."""
    subsets = {
        "rave_effect": [
            "Random",
            "MCTS (default)",
            "MCTS (no RAVE)",
            "MCTS (heavy RAVE)",
        ],
        "pw_effect": [
            "Random",
            "MCTS (default)",
            "MCTS (no PW)",
            "MCTS (tight PW)",
            "MCTS (loose PW)",
        ],
        "exploration": [
            "Random",
            "MCTS (default)",
            "MCTS (low explore)",
            "MCTS (high explore)",
        ],
        "p_stop": [
            "Random",
            "MCTS (default)",
            "MCTS (p_stop=0.1)",
            "MCTS (p_stop=0.6)",
            "MCTS (adaptive p)",
            "MCTS (no RAVE+adpt)",
            "MCTS (norm)",
            "MCTS (no RAVE+adpt+norm)",
        ],
        "rollout": [
            "Random",
            "MCTS (no RAVE+adpt+norm)",
            "MCTS (+rpol)",
            "MCTS (+rpol ε=0.1)",
            "MCTS (+rpol τ=0.5)",
            "MCTS (+rpol τ=2)",
        ],
        "crave": [
            "Random",
            "MCTS (no RAVE+adpt+norm)",
            "MCTS (+rpol)",
            "MCTS (+crave k=100)",
            "MCTS (+crave k=300)",
            "MCTS (+crave k=500)",
        ],
    }
    for subset_name, cnames in subsets.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for cname in cnames:
            key = (problem_name, cname)
            if key not in all_results:
                continue
            results = all_results[key]
            curves = np.array([r.best_values_over_time for r in results])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            x = np.arange(1, n_iterations + 1)
            color = COLOR_MAP.get(cname, None)
            ax.plot(x, mean, label=cname, color=color, linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=color)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Best Reward Found", fontsize=12)
        ax.set_title(
            f"{problem_name} — {subset_name.replace('_', ' ').title()}", fontsize=13
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = OUTPUT_DIR / f"convergence_{problem_name}_{subset_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


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
            prob.optimal_value, color="red", linestyle="--", alpha=0.5, label="Optimum"
        )
        ax.legend(fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Final Best Reward by Configuration", fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "summary_bar_chart.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_optimum_rate_heatmap(
    problems: list[Problem], all_results: dict, config_names: list[str]
):
    """Heatmap: optimum-finding rate (config × problem)."""
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

    # Annotate cells
    for i in range(len(config_names)):
        for j in range(len(problems)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(
                j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color
            )

    ax.set_title("Optimum-Finding Rate", fontsize=13)
    fig.colorbar(im, ax=ax, label="Rate", shrink=0.8)
    fig.tight_layout()
    path = OUTPUT_DIR / "optimum_rate_heatmap.png"
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

    fig.suptitle("Exploration: Unique Selections Evaluated", fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUTPUT_DIR / "unique_evals.png"
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

    path = OUTPUT_DIR / "results.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    print("MCTS Benchmark Suite")
    print("=" * 70)

    problems = [
        make_problem_multigroup_interaction(),
        make_problem_needle(),
        make_problem_mixed(),
        make_problem_large_sparse(),
        make_problem_graduated(),
    ]

    for p in problems:
        print(
            f"  {p.name}: ~{p.search_space_size:,} combinations, {p.n_iterations} iters × {p.n_trials} trials"
        )

    all_config_names = ["Random"] + [c.name for c in CONFIGS]

    t_start = time.time()
    all_results = run_benchmark(problems, CONFIGS)
    total_time = time.time() - t_start
    print(f"\nTotal benchmark time: {total_time:.1f}s")

    # Generate plots
    print("\nGenerating plots...")
    for prob in problems:
        plot_convergence_curves(
            prob.name, all_results, all_config_names, prob.n_iterations
        )
        plot_convergence_subsets(prob.name, all_results, prob.n_iterations)

    plot_summary_bar_chart(problems, all_results, all_config_names)
    plot_optimum_rate_heatmap(problems, all_results, all_config_names)
    plot_unique_evals(problems, all_results, all_config_names)

    # Save numeric results
    summary = save_summary_json(problems, all_results, all_config_names)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    for prob in problems:
        print(
            f"\n{prob.name} (search space: ~{prob.search_space_size:,}, optimum: {prob.optimal_value})"
        )
        print(
            f"  {'Config':<25s} {'Mean Best':>10s} {'±Std':>8s} {'Opt Rate':>10s} {'Uniq Evals':>12s}"
        )
        print(f"  {'-'*65}")
        for cname in all_config_names:
            d = summary[prob.name].get(cname)
            if d is None:
                continue
            print(
                f"  {cname:<25s} {d['mean_best']:10.1f} {d['std_best']:8.1f} "
                f"{d['optimum_rate']:10.0%} {d['mean_unique_evals']:12.0f}"
            )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
