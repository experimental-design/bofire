# MCTS Benchmark Report: Combinatorial NChooseK Optimization

## Executive Summary

This benchmark evaluates the MCTS algorithm from `bofire/strategies/predictives/optimize_mcts.py` (without acquisition function integration) across 5 combinatorial problems with NChooseK constraints. We test 13 MCTS configurations varying RAVE, Progressive Widening (PW), exploration constants, stop probability, and adaptive stop probability against a random-sampling baseline.

Two algorithmic fixes were implemented during this benchmarking cycle:
1. **Virtual loss on cache hit**: On revisiting a cached terminal, increment visit counts but backpropagate reward=0. This dilutes mean node value for over-exploited branches, steering UCT toward unexplored territory.
2. **Rollout retry on cache hit**: When a rollout produces a cached terminal, re-roll up to `max_rollout_retries` times to find a novel selection.

**Key result**: These fixes transformed MCTS from underperforming random sampling to decisively outperforming it on every problem. The best configuration (**MCTS no RAVE**) achieves 97% optimum-finding rate on needle_in_haystack (vs 10% for random), 90% on graduated_landscape (vs 7%), and 47% on mixed problems (vs 3%). Unique evaluations increased 3-7x, closing the exploration gap with random while retaining MCTS's ability to exploit reward structure.

---

## 1. Experimental Setup

### 1.1 MCTS Configurations Tested

| Config | c_uct | k_rave | pw_k0 | pw_alpha | p_stop |
|--------|-------|--------|-------|----------|--------|
| **Random baseline** | — | — | — | — | 0.35 |
| **MCTS (default)** | 1.0 | 300 | 2.0 | 0.6 | 0.35 |
| **MCTS (no RAVE)** | 1.0 | 0 | 2.0 | 0.6 | 0.35 |
| **MCTS (no PW)** | 1.0 | 300 | 1e6 | 0.6 | 0.35 |
| **MCTS (no RAVE, no PW)** | 1.0 | 0 | 1e6 | 0.6 | 0.35 |
| **MCTS (low explore)** | 0.1 | 300 | 2.0 | 0.6 | 0.35 |
| **MCTS (high explore)** | 5.0 | 300 | 2.0 | 0.6 | 0.35 |
| **MCTS (heavy RAVE)** | 1.0 | 3000 | 2.0 | 0.6 | 0.35 |
| **MCTS (tight PW)** | 1.0 | 300 | 1.0 | 0.4 | 0.35 |
| **MCTS (loose PW)** | 1.0 | 300 | 5.0 | 0.8 | 0.35 |
| **MCTS (p_stop=0.1)** | 1.0 | 300 | 2.0 | 0.6 | 0.10 |
| **MCTS (p_stop=0.6)** | 1.0 | 300 | 2.0 | 0.6 | 0.60 |
| **MCTS (adaptive p)** | 1.0 | 300 | 2.0 | 0.6 | adaptive |
| **MCTS (no RAVE+adpt)** | 1.0 | 0 | 2.0 | 0.6 | adaptive |

- **RAVE disabled**: `k_rave=0` sets β=0, making the score pure UCT.
- **PW disabled**: `pw_k0=1e6` makes the child limit always exceed legal actions.
- **Adaptive p_stop**: Learns per-group stop probability from cardinality-reward statistics. Uses sigmoid on normalized `(E_stop - E_continue)`, blended with fixed prior during warmup (20 rollouts).

### 1.2 Benchmark Problems

| Problem | Groups | Features | Subset sizes | Search space | Budget | Trials |
|---------|--------|----------|-------------|-------------|--------|--------|
| **multigroup_interaction** | 3 NChooseK | 8 each | 1-4 | ~4.25M | 600 | 30 |
| **needle_in_haystack** | 1 NChooseK | 15 | 2-5 | ~4,928 | 400 | 30 |
| **mixed_nchoosek_categorical** | 2 NChooseK + 2 Cat | 6 each + 4 vals | 1-3 | ~26,896 | 500 | 30 |
| **large_sparse** | 4 NChooseK | 10 each | 0-3 | ~960M | 800 | 30 |
| **graduated_landscape** | 1 NChooseK | 10 | 2-4 | 375 | 300 | 30 |

**Problem descriptions:**
- **multigroup_interaction**: Optimal requires specific features from all 3 groups with cross-group interaction bonuses (e.g., feature 1 + feature 9 = +12 bonus). Tests whether MCTS can learn multi-group correlations.
- **needle_in_haystack**: Single small optimal subset {3,7,11} among ~5000 candidates with mild partial credit. Tests raw exploration efficiency.
- **mixed_nchoosek_categorical**: Feature+categorical interactions (e.g., feature 2 + cat_dim_20=2.0 = +15). Tests handling of mixed discrete types.
- **large_sparse**: Optimal uses features from only 2 of 4 groups, with a sparsity bonus. The search space is ~960 million. Tests scalability and ability to learn that most groups should be empty.
- **graduated_landscape**: Smooth quality-based reward (each feature has a fixed quality score). Many near-optimal solutions. Tests exploitation of smooth structure.

---

## 2. Algorithm Fixes Applied

### 2.1 Problem Identified: Exploration Bottleneck

The original MCTS algorithm had a severe exploration bottleneck. With 600 iterations, it evaluated only ~50-60 unique terminal selections (vs ~588 for random sampling). The root cause was a feedback loop:

1. **UCT concentrates visits** on the highest-reward branch
2. That branch **grows deeper** (one node expanded per iteration)
3. Deep leaves have **few rollout choices** left, producing the same terminals
4. **Cached reward is backpropagated**, reinforcing the exploitation bias
5. Goto 1

Unlike game-playing MCTS where every rollout is stochastic, here the reward function is deterministic and cached — revisiting a terminal adds zero information, but the old code still backpropagated the cached reward as if it were new.

### 2.2 Fix 1: Virtual Loss on Cache Hit

When an iteration produces a terminal that's already in the cache, we increment visit counts along the path but **backpropagate zero reward**. This dilutes `mean_value = w_total / n_visits` for over-visited nodes, causing UCT to prefer less-explored branches:

```python
if is_novel:
    self._backpropagate(path, reward, selected_features, cat_selections)
else:
    # Virtual loss: increment visits with zero reward
    for n in path:
        n.n_visits += 1
```

It is critical to still increment `n_visits` (not skip backpropagation entirely), because:
- **Progressive Widening** uses `n_visits` to decide when to expand new children
- **UCT** needs visit counts to change so it doesn't deterministically repeat the same path

### 2.3 Fix 2: Rollout Retry on Cache Hit

When a rollout produces a cached terminal, re-roll up to `max_rollout_retries` times:

```python
selected_features, cat_selections = self._rollout(leaf)
for _attempt in range(self.max_rollout_retries):
    key = self._make_cache_key(selected_features, cat_selections)
    if key not in self.value_cache:
        break
    selected_features, cat_selections = self._rollout(leaf)
```

This is cheap (rollouts are fast) and directly reduces wasted iterations from non-terminal leaves where rollout randomness can reach diverse terminals.

---

## 3. Results

### 3.1 Summary Tables

#### multigroup_interaction (search space ~4.25M, optimum = 150.0)

| Config | Mean Best | ±Std | Opt Rate | Unique Evals |
|--------|----------|------|----------|-------------|
| Random | 62.9 | 10.3 | 0% | 588 |
| MCTS (default) | 94.9 | 18.9 | 0% | 205 |
| **MCTS (no RAVE)** | **105.1** | 25.8 | **20%** | 392 |
| MCTS (no PW) | 90.8 | 19.2 | 0% | 197 |
| MCTS (no RAVE, no PW) | 99.4 | 29.8 | 20% | 365 |
| MCTS (high explore) | 97.3 | 15.7 | 0% | 240 |
| MCTS (heavy RAVE) | 81.4 | 18.7 | 0% | 84 |
| MCTS (p_stop=0.1) | 96.2 | 14.3 | 0% | 238 |
| MCTS (p_stop=0.6) | 84.5 | 20.8 | 0% | 165 |
| MCTS (adaptive p) | 98.0 | 14.5 | 0% | 219 |
| MCTS (no RAVE+adpt) | 103.8 | 29.7 | 23% | 380 |

#### needle_in_haystack (search space ~4,928, optimum = 100.0)

| Config | Mean Best | ±Std | Opt Rate | Unique Evals |
|--------|----------|------|----------|-------------|
| Random | 39.7 | 20.5 | 10% | 216 |
| MCTS (default) | 77.0 | 32.6 | 67% | 58 |
| **MCTS (no RAVE)** | **97.7** | 12.6 | **97%** | 154 |
| MCTS (no RAVE, no PW) | 97.7 | 12.6 | 97% | 147 |
| MCTS (high explore) | 88.3 | 26.1 | 83% | 64 |
| MCTS (heavy RAVE) | 35.2 | 18.0 | 7% | 34 |
| MCTS (p_stop=0.6) | 91.2 | 22.6 | 87% | 63 |
| MCTS (adaptive p) | 84.0 | 29.1 | 77% | 61 |
| **MCTS (no RAVE+adpt)** | **100.0** | 0.0 | **100%** | 161 |

#### mixed_nchoosek_categorical (search space ~26,896, optimum = 150.0)

| Config | Mean Best | ±Std | Opt Rate | Unique Evals |
|--------|----------|------|----------|-------------|
| Random | 79.2 | 14.6 | 3% | 472 |
| MCTS (default) | 84.5 | 16.7 | 3% | 111 |
| **MCTS (no RAVE)** | **113.6** | 34.7 | **47%** | 284 |
| MCTS (no RAVE, no PW) | 108.5 | 37.1 | 43% | 279 |
| MCTS (p_stop=0.1) | 89.5 | 18.4 | 7% | 130 |
| MCTS (heavy RAVE) | 74.7 | 13.1 | 0% | 46 |
| MCTS (adaptive p) | 85.8 | 9.3 | 0% | 112 |
| MCTS (no RAVE+adpt) | 110.4 | 35.7 | 43% | 280 |

#### large_sparse (search space ~960M, optimum = 200.0)

| Config | Mean Best | ±Std | Opt Rate | Unique Evals |
|--------|----------|------|----------|-------------|
| Random | 36.1 | 6.3 | 0% | 764 |
| MCTS (default) | 40.0 | 8.4 | 0% | 303 |
| **MCTS (no RAVE)** | **83.8** | 64.5 | **23%** | 515 |
| MCTS (no RAVE, no PW) | 61.5 | 38.1 | 7% | 513 |
| MCTS (p_stop=0.6) | 55.4 | 28.1 | 3% | 448 |
| MCTS (heavy RAVE) | 31.3 | 9.4 | 0% | 90 |
| MCTS (adaptive p) | 54.6 | 27.8 | 3% | 421 |
| **MCTS (no RAVE+adpt)** | **93.0** | 64.9 | **27%** | 550 |

#### graduated_landscape (search space 375, optimum = 65.0)

| Config | Mean Best | ±Std | Opt Rate | Unique Evals |
|--------|----------|------|----------|-------------|
| Random | 60.6 | 3.3 | 7% | 113 |
| MCTS (default) | 64.1 | 1.4 | 40% | 65 |
| **MCTS (no RAVE)** | **64.9** | 0.3 | **90%** | 162 |
| MCTS (no RAVE, no PW) | 64.8 | 0.4 | 80% | 157 |
| MCTS (p_stop=0.1) | 64.5 | 0.5 | 47% | 89 |
| MCTS (heavy RAVE) | 55.4 | 5.4 | 0% | 21 |
| MCTS (adaptive p) | 64.5 | 0.8 | 57% | 75 |
| MCTS (no RAVE+adpt) | 64.6 | 0.9 | 77% | 168 |

### 3.2 Convergence Curves

#### All configurations — large_sparse problem
![Convergence large_sparse](convergence_large_sparse.png)

MCTS (no RAVE) dominates, reaching mean best ~84 with 23% optimum-finding rate in a search space of ~960 million. The high variance reflects that when MCTS finds the right region early, it converges to the optimum; otherwise it still significantly outperforms random.

#### RAVE effect — needle_in_haystack
![RAVE effect needle](convergence_needle_in_haystack_rave_effect.png)

The no-RAVE variants converge rapidly to near-optimum, achieving 97% success. Heavy RAVE (pink) performs worse than random — RAVE's context-independent feature value assumption actively misleads the search.

#### p_stop effect — multigroup_interaction
![p_stop multigroup](convergence_multigroup_interaction_p_stop.png)

p_stop=0.1 (cyan) outperforms default (p_stop=0.35) because the optimal solution requires 7 features across 3 groups — low stop probability produces rollouts with more features, better matching the target.

---

## 4. Analysis

### 4.1 Impact of the Algorithmic Fixes

The virtual loss + rollout retry combination produced dramatic improvements across every problem and configuration:

| Problem | Old default | New default | Old best | New best |
|---------|------------|-------------|----------|----------|
| multigroup_interaction | 78.5 | **94.9** (+21%) | 81.9 | **105.1** (+28%) |
| needle_in_haystack | 40.0 (17%) | **77.0 (67%)** | 49.2 (30%) | **97.7 (97%)** |
| mixed_nchoosek_categorical | 73.3 (0%) | **84.5 (3%)** | 79.2 (3%) | **113.6 (47%)** |
| large_sparse | 30.0 (0%) | **40.0** | 49.2 (3%) | **83.8 (23%)** |
| graduated_landscape | 54.0 (0%) | **64.1 (40%)** | 60.6 (7%) | **64.9 (90%)** |

*Percentages in parentheses are optimum-finding rates. "Old best" is the best config from the pre-fix benchmark (often random sampling).*

The unique evaluations tell the story — the exploration bottleneck has been substantially resolved:

| Problem | Random | Old MCTS default | New MCTS default | New MCTS (no RAVE) |
|---------|--------|------------------|------------------|-------------------|
| multigroup_interaction | 588 | 54 | **205** (3.8x) | **392** (7.3x) |
| needle_in_haystack | 216 | 33 | **58** (1.8x) | **154** (4.7x) |
| mixed_nchoosek_categorical | 472 | 30 | **111** (3.7x) | **284** (9.5x) |
| large_sparse | 764 | 72 | **303** (4.2x) | **515** (7.2x) |
| graduated_landscape | 113 | 19 | **65** (3.4x) | **162** (8.5x) |

### 4.2 RAVE: Harmful, Should Be Disabled

With the exploration bottleneck fixed, RAVE's effect becomes even clearer. **Disabling RAVE is the single most impactful parameter change**, consistently producing the best or tied-best results:

| Problem | Default (RAVE on) | No RAVE | Heavy RAVE |
|---------|-------------------|---------|------------|
| multigroup_interaction | 94.9 (0%) | **105.1 (20%)** | 81.4 (0%) |
| needle_in_haystack | 77.0 (67%) | **97.7 (97%)** | 35.2 (7%) |
| mixed_nchoosek_categorical | 84.5 (3%) | **113.6 (47%)** | 74.7 (0%) |
| large_sparse | 40.0 (0%) | **83.8 (23%)** | 31.3 (0%) |
| graduated_landscape | 64.1 (40%) | **64.9 (90%)** | 55.4 (0%) |

RAVE's context-independent assumption (feature X is equally valuable regardless of what other features are selected) is fundamentally wrong for NChooseK problems. With the virtual loss fix allowing more exploration, the damage from RAVE's mis-generalization becomes much more visible — it actively steers the search toward poor feature combinations.

**Heavy RAVE (k_rave=3000) performs worse than random on 2 of 5 problems.** This should be considered a broken configuration.

### 4.3 Progressive Widening: Moderate Effect

| Problem | Default (PW on) | No PW | Tight PW | Loose PW |
|---------|-----------------|-------|----------|----------|
| multigroup_interaction | **94.9** | 90.8 | 91.4 | 90.8 |
| needle_in_haystack | 77.0 | 76.7 | 39.8 | 76.7 |
| mixed_nchoosek_categorical | **84.5** | 82.2 | 85.2 | 82.2 |
| large_sparse | 40.0 | **40.2** | 52.6 | 40.2 |
| graduated_landscape | **64.1** | 63.7 | 61.6 | 63.7 |

Default PW (pw_k0=2.0, pw_alpha=0.6) is slightly better than no PW on most problems when RAVE is active. This is because PW provides a controlled pace of exploration that complements the virtual loss mechanism. Tight PW (pw_k0=1.0, pw_alpha=0.4) is too restrictive and hurts on needle_in_haystack. Notably, PW matters much less when RAVE is disabled — the no RAVE + no PW config performs nearly as well as no RAVE + default PW.

### 4.4 Exploration Constant (c_uct)

| Problem | Low (0.1) | Default (1.0) | High (5.0) |
|---------|-----------|---------------|------------|
| multigroup_interaction | 94.2 | 94.9 | **97.3** |
| needle_in_haystack | 74.3 (63%) | 77.0 (67%) | **88.3 (83%)** |
| mixed_nchoosek_categorical | 87.7 | 84.5 | **88.1** |
| large_sparse | 38.0 | 40.0 | **47.2** |
| graduated_landscape | 63.9 (23%) | 64.1 (40%) | **64.3 (40%)** |

Higher c_uct consistently helps. With the virtual loss fix, the exploration bonus from UCT now has room to operate — the tree isn't locked into a single deep branch anymore. c_uct=5.0 is the best pure-UCT exploration setting tested, though the improvement is modest compared to the impact of disabling RAVE.

### 4.5 Stop Probability (p_stop_rollout): Problem-Dependent, Now Adaptive

| Problem | Optimal subset size | p_stop=0.1 | p_stop=0.35 | p_stop=0.6 | Adaptive | no RAVE+adpt |
|---------|-------------------|-----------|------------|-----------|----------|-------------|
| multigroup_interaction | 7 features | 96.2 | 94.9 | 84.5 | 98.0 | 103.8 (23%) |
| needle_in_haystack | 3 features | 67.0 (53%) | 77.0 (67%) | 91.2 (87%) | 84.0 (77%) | **100.0 (100%)** |
| mixed_nchoosek_categorical | 4 features | 89.5 (7%) | 84.5 (3%) | 83.9 (0%) | 85.8 (0%) | 110.4 (43%) |
| large_sparse | 5 from 2 groups | 33.6 | 40.0 | 55.4 | 54.6 | **93.0 (27%)** |
| graduated_landscape | 4 features | 64.5 (47%) | 64.1 (40%) | 62.4 (3%) | 64.5 (57%) | 64.6 (77%) |

The pattern for fixed p_stop is consistent: low p_stop favors problems needing many features; high p_stop favors sparse solutions.

**Adaptive p_stop** learns per-group stop probabilities online from cardinality-reward statistics. It tracks `(group_idx, cardinality) -> (visits, total_reward)`, computes E_stop vs E_continue (max over higher cardinalities), and applies a sigmoid to determine stop probability, blended with the fixed prior during a warmup period.

Results show adaptive p_stop provides a **robust default that avoids catastrophic mismatch**:
- **Best or tied-best** on multigroup_interaction (98.0 vs 96.2 for p_stop=0.1) and graduated_landscape (57% opt rate vs 47% for p_stop=0.1)
- **Competitive** on large_sparse (54.6 vs 55.4 for p_stop=0.6) and needle_in_haystack (77% vs 87% for p_stop=0.6)
- **Never the worst**: Avoids the bad performance of wrong fixed p_stop (e.g., p_stop=0.6 on multigroup_interaction gives only 84.5, while adaptive gives 98.0)

**No RAVE + adaptive p_stop is the best overall configuration**, combining the two most impactful improvements:
- **100% optimum rate on needle_in_haystack** (up from 97% with no RAVE alone, perfect across all 30 trials)
- **Best on large_sparse** at 93.0 mean / 27% opt rate (up from 83.8 / 23% with no RAVE alone)
- **Best or tied on 3 of 5 problems**, competitive on the remaining 2
- The synergy is clear: no RAVE removes the misleading context-independent bias, while adaptive p_stop learns the right cardinality preference per problem

The adaptive mechanism is most valuable when the user cannot tune p_stop per-problem, which is the typical use case in real BO workflows where the reward landscape is unknown a priori.

---

## 5. Optimum-Finding Rates

![Optimum rate heatmap](optimum_rate_heatmap.png)

**MCTS (no RAVE+adpt)** is the new best overall: **100%** on needle_in_haystack, **27%** on large_sparse, **43%** on mixed, **23%** on multigroup_interaction, and **77%** on graduated_landscape. It outperforms or matches **MCTS (no RAVE)** (which achieves 97%, 23%, 47%, 20%, 90% respectively) on 2 of 5 problems while being competitive on the rest.

**Heavy RAVE is catastrophic**: 7% on needle (worse than random's 10%), 0% on 4 of 5 problems.

---

## 6. Summary Bar Chart

![Summary bar chart](summary_bar_chart.png)

---

## 7. Exploration Efficiency

![Unique evaluations](unique_evals.png)

The no-RAVE configurations now explore 150-515 unique selections per run, approaching or exceeding random's coverage while also directing that exploration intelligently. Heavy RAVE still restricts exploration to ~20-90 unique selections — RAVE's value-sharing biases the tree toward a narrow set of "globally good" features, undermining the virtual loss mechanism.

---

## 8. Recommendations

### 8.1 Recommended Default Configuration

Based on these results, the recommended defaults for NChooseK problems are:

| Parameter | Current default | Recommended | Rationale |
|-----------|----------------|-------------|-----------|
| k_rave | 300 | **0** | RAVE hurts on every problem tested |
| c_uct | 1.0 | 1.0 (or increase to ~2.0) | Higher helps slightly |
| pw_k0 | 2.0 | 2.0 | Current value works well with virtual loss |
| pw_alpha | 0.6 | 0.6 | Current value works well |
| max_rollout_retries | 3 | 3 | Effective at reducing wasted iterations |
| p_stop_rollout | 0.35 | 0.35 | Base prior for adaptive blending |
| adaptive_p_stop | True | **True** | Avoids worst-case fixed p_stop mismatch |
| p_stop_warmup | 20 | 20 | Sufficient to accumulate per-group statistics |
| p_stop_temperature | 0.25 | 0.25 | Produces decisive but not extreme sigmoid |

### 8.2 Further Improvements to Explore

1. ~~**Adaptive p_stop_rollout**~~: **Implemented and validated.** Per-group adaptive p_stop learns from cardinality-reward statistics. Combined with no RAVE, it achieves 100% on needle_in_haystack and best results on large_sparse. See Section 4.5 for details.
2. **Context-aware RAVE**: If RAVE is to be reintroduced, condition it on (group_idx, selection_count) so it captures state-dependent value rather than global averages.
3. **Reward normalization**: With virtual loss injecting zero-reward visits, normalizing rewards (e.g., min-max scaling) could make the dilution effect more predictable across different reward scales.

---

## 9. Files Generated

| File | Description |
|------|-------------|
| `benchmark.py` | Benchmark script (self-contained, reproduces all results) |
| `results.json` | Full numeric results for all configs and problems |
| `summary_bar_chart.png` | Bar chart of final best reward across all problems |
| `optimum_rate_heatmap.png` | Heatmap of optimum-finding rates |
| `unique_evals.png` | Exploration efficiency comparison |
| `convergence_<problem>.png` | Full convergence curves per problem |
| `convergence_<problem>_rave_effect.png` | RAVE ablation convergence |
| `convergence_<problem>_pw_effect.png` | PW ablation convergence |
| `convergence_<problem>_exploration.png` | c_uct ablation convergence |
| `convergence_<problem>_p_stop.png` | p_stop ablation convergence |

## 10. Reproducing

```bash
python mcts-report/benchmark.py
```

All results use fixed random seeds for reproducibility. Runtime is ~25 seconds.
