"""MCTS with Thompson Sampling for NChooseK and categorical optimization.

Replaces UCT tree policy and softmax rollout policy with Bayesian posteriors
(Normal-Normal conjugate update), eliminating most tunable hyperparameters.

Tree selection: sample from each child's Normal posterior, pick highest.
Rollout modes: uniform, TS per (group,action), TS per (group,cardinality,action),
or softmax fallback (pure-math, no torch).

Bayesian update (weak prior, estimated variance):
    Prior: N(mu0, sigma0^2), mu0 = running global mean, n0 = 1
        sigma0^2 = ts_prior_var (fixed) or empirical reward variance (adaptive)
    After n observations:
        x_bar = sum_rewards / n
        s^2 = max(sum_sq_rewards/n - x_bar^2, 1e-8)
        posterior_mean = (n0*mu0 + n*x_bar) / (n0 + n)
        posterior_var  = s^2 / (n0 + n)
        sample ~ N(posterior_mean, posterior_var)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

from optimize_mcts_full import STOP, Groups


# =============================================================================
# TS-specific types
# =============================================================================


class TSActionStats(NamedTuple):
    """Sufficient statistics for Thompson Sampling on a rollout action."""

    n_obs: int
    sum_rewards: float
    sum_sq_rewards: float


@dataclass
class TSNode:
    """MCTS tree node with Thompson Sampling statistics.

    Replaces UCT's single n_visits/w_total with Bayesian sufficient statistics.

    Args:
        partial_by_group: Partial selection per group (indices into group's options)
        stopped_by_group: Whether each group has stopped selecting (for NChooseK)
        group_idx: Current group being filled
        n_obs: Novel observations count (used for posterior updates)
        sum_rewards: Sum of observed rewards from novel evaluations
        sum_sq_rewards: Sum of squared rewards from novel evaluations
        n_visits: Total visits including cache hits (for progressive widening)
        children: Child nodes keyed by action (int index or STOP)
    """

    partial_by_group: tuple[tuple[int, ...], ...]
    stopped_by_group: tuple[bool, ...]
    group_idx: int

    n_obs: int = 0
    sum_rewards: float = 0.0
    sum_sq_rewards: float = 0.0
    n_visits: int = 0

    children: dict[int, "TSNode"] = field(default_factory=dict)

    def is_terminal(self, groups: Groups) -> bool:
        return self.group_idx >= len(groups)

    def mean_value(self) -> float:
        return self.sum_rewards / self.n_obs if self.n_obs > 0 else 0.0


# =============================================================================
# MCTS with Thompson Sampling
# =============================================================================


class MCTS_TS:
    """Monte Carlo Tree Search with Thompson Sampling.

    Uses Normal-Normal conjugate posteriors for both tree selection and
    (optionally) rollout action selection, replacing UCT and softmax policies.

    Args:
        groups: Collection of NChooseK and categorical constraints
        reward_fn: Function mapping (selected_features, cat_selections) to reward
        ts_prior_var: Prior variance for Normal posterior (default 1.0); used as
            fallback when adaptive_prior_var=True and fewer than 2 rewards observed
        adaptive_prior_var: If True, set prior variance to the running empirical
            variance of all observed rewards instead of the fixed ts_prior_var
        cache_hit_mode: How to handle cache hits: "no_update" or "variance_inflation"
        variance_decay: Decay factor for variance inflation mode (default 0.95)
        rollout_mode: Rollout policy: "uniform", "ts_group_action",
            "ts_group_card_action", or "softmax"
        pw_k0: Progressive widening base constant (default 2.0)
        pw_alpha: Progressive widening exponent (default 0.6)
        max_rollout_retries: Maximum rollout retries on cache hit (default 3)
        seed: Random seed for reproducibility
        rollout_epsilon: Epsilon-mix for uniform exploration in softmax mode (default 0.3)
        rollout_tau: Softmax temperature in softmax mode (default 1.0)
        rollout_novelty_weight: Novelty bonus coefficient in softmax mode (default 1.0)
        normalize_rewards: Normalize rewards for softmax rollout stats (default True)
        adaptive_p_stop: Enable adaptive stop probability in softmax mode (default True)
        p_stop_rollout: Base stop probability in softmax mode (default 0.35)
        p_stop_warmup: Warmup count for adaptive p_stop (default 20)
        p_stop_temperature: Sigmoid temperature for adaptive p_stop (default 0.25)
    """

    def __init__(
        self,
        groups: Groups,
        reward_fn: Callable[[tuple[int, ...], dict[int, float]], float],
        ts_prior_var: float = 1.0,
        adaptive_prior_var: bool = False,
        cache_hit_mode: str = "no_update",
        variance_decay: float = 0.95,
        rollout_mode: str = "uniform",
        pw_k0: float = 2.0,
        pw_alpha: float = 0.6,
        max_rollout_retries: int = 3,
        seed: Optional[int] = None,
        # Softmax fallback parameters
        rollout_epsilon: float = 0.3,
        rollout_tau: float = 1.0,
        rollout_novelty_weight: float = 1.0,
        normalize_rewards: bool = True,
        adaptive_p_stop: bool = True,
        p_stop_rollout: float = 0.35,
        p_stop_warmup: int = 20,
        p_stop_temperature: float = 0.25,
    ):
        self.groups = groups
        self.reward_fn = reward_fn
        self.ts_prior_var = ts_prior_var
        self.adaptive_prior_var = adaptive_prior_var
        self.cache_hit_mode = cache_hit_mode
        self.variance_decay = variance_decay
        self.rollout_mode = rollout_mode
        self.pw_k0 = pw_k0
        self.pw_alpha = pw_alpha
        self.max_rollout_retries = max_rollout_retries
        self.rng = random.Random(seed)

        # Softmax fallback params
        self.rollout_epsilon = rollout_epsilon
        self.rollout_tau = rollout_tau
        self.rollout_novelty_weight = rollout_novelty_weight
        self.normalize_rewards = normalize_rewards
        self.adaptive_p_stop = adaptive_p_stop
        self.p_stop_rollout = p_stop_rollout
        self.p_stop_warmup = p_stop_warmup
        self.p_stop_temperature = p_stop_temperature

        # Initialize root node
        n_groups = len(groups)
        self.root = TSNode(
            partial_by_group=tuple(() for _ in range(n_groups)),
            stopped_by_group=tuple(False for _ in range(n_groups)),
            group_idx=0,
        )

        # Best found so far
        self.best_selection: Optional[tuple[tuple[int, ...], dict[int, float]]] = None
        self.best_value: float = float("-inf")

        # Cache for terminal evaluations
        self.value_cache: dict[tuple, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Global reward tracking for TS prior center and adaptive variance
        self._novel_reward_sum: float = 0.0
        self._novel_reward_sq_sum: float = 0.0
        self._novel_reward_count: int = 0

        # Rollout TS statistics: key -> TSActionStats
        # Key format depends on rollout_mode:
        #   ts_group_action: (group_idx, action)
        #   ts_group_card_action: (group_idx, cardinality, action)
        self.rollout_ts_stats: dict[tuple, TSActionStats] = {}

        # Softmax rollout statistics (for softmax fallback)
        self.rollout_stats: dict[tuple[int, int], tuple[int, float]] = {}

        # Adaptive p_stop statistics (for softmax fallback)
        self.cardinality_stats: dict[tuple[int, int], tuple[int, float]] = {}
        n_nchoosek = len(self.groups.nchooseks)
        self.group_rollout_counts: list[int] = [0] * n_nchoosek
        self.reward_min: float = float("inf")
        self.reward_max: float = float("-inf")

    # ─── Prior center ───────────────────────────────────────────────

    def _global_mean(self) -> float:
        """Running mean of all novel rewards (prior center mu0)."""
        if self._novel_reward_count == 0:
            return 0.0
        return self._novel_reward_sum / self._novel_reward_count

    def _prior_var(self) -> float:
        """Prior variance sigma0^2, either fixed or adaptive.

        When adaptive_prior_var=True, returns the running empirical variance
        of all novel rewards once at least 2 observations exist. This
        auto-calibrates the prior to the problem's reward scale — the TS
        analogue of reward normalization for UCT.
        """
        if not self.adaptive_prior_var or self._novel_reward_count < 2:
            return self.ts_prior_var
        mean = self._global_mean()
        empirical_var = (
            self._novel_reward_sq_sum / self._novel_reward_count - mean * mean
        )
        return max(empirical_var, 1e-8)

    # ─── Thompson Sampling ──────────────────────────────────────────

    def _ts_sample_score(self, node: TSNode) -> float:
        """Sample from node's Normal posterior for tree selection."""
        mu0 = self._global_mean()
        sigma0_sq = self._prior_var()
        n0 = 1  # pseudo-count

        n = node.n_obs
        if n == 0:
            return self.rng.gauss(mu0, math.sqrt(sigma0_sq))

        x_bar = node.sum_rewards / n
        s_sq = max(node.sum_sq_rewards / n - x_bar * x_bar, 1e-8)

        post_mean = (n0 * mu0 + n * x_bar) / (n0 + n)
        post_var = s_sq / (n0 + n)

        return self.rng.gauss(post_mean, math.sqrt(post_var))

    def _ts_sample_action_score(self, stats: TSActionStats) -> float:
        """Sample from a TSActionStats posterior (for rollout actions)."""
        mu0 = self._global_mean()
        sigma0_sq = self._prior_var()
        n0 = 1

        n = stats.n_obs
        if n == 0:
            return self.rng.gauss(mu0, math.sqrt(sigma0_sq))

        x_bar = stats.sum_rewards / n
        s_sq = max(stats.sum_sq_rewards / n - x_bar * x_bar, 1e-8)

        post_mean = (n0 * mu0 + n * x_bar) / (n0 + n)
        post_var = s_sq / (n0 + n)

        return self.rng.gauss(post_mean, math.sqrt(post_var))

    # ─── Copied from production (Node -> TSNode) ────────────────────

    def _make_cache_key(
        self, selected_features: tuple[int, ...], cat_selections: dict[int, float]
    ) -> tuple:
        """Create hashable cache key from selection."""
        return (selected_features, frozenset(cat_selections.items()))

    def _cached_reward(
        self, selected_features: tuple[int, ...], cat_selections: dict[int, float]
    ) -> float:
        """Get cached reward or compute and cache it."""
        key = self._make_cache_key(selected_features, cat_selections)
        if key in self.value_cache:
            self.cache_hits += 1
            return self.value_cache[key]
        val = self.reward_fn(selected_features, cat_selections)
        self.value_cache[key] = val
        self.cache_misses += 1
        return val

    def _child_limit(self, node: TSNode) -> int:
        """Progressive widening: max children based on visit count."""
        return max(1, int(self.pw_k0 * (max(1, node.n_visits) ** self.pw_alpha)))

    def _legal_actions(self, node: TSNode) -> list[int]:
        """Get legal actions for current group in node."""
        if node.is_terminal(self.groups):
            return []
        g = node.group_idx
        group = self.groups.groups[g]
        partial = node.partial_by_group[g]
        stopped = node.stopped_by_group[g]
        return group.legal_actions(partial, stopped)

    def _apply_action(self, node: TSNode, action: int) -> TSNode:
        """Create child node by applying action to current node."""
        g = node.group_idx
        group = self.groups.groups[g]

        partials = list(node.partial_by_group)
        stoppeds = list(node.stopped_by_group)

        if action == STOP:
            stoppeds[g] = True
            next_g = g + 1
        else:
            partials[g] += (action,)
            if group.is_complete(partials[g], stoppeds[g]):
                next_g = g + 1
            else:
                next_g = g

        return TSNode(
            partial_by_group=tuple(partials),
            stopped_by_group=tuple(stoppeds),
            group_idx=next_g,
        )

    def _get_selection(self, node: TSNode) -> tuple[tuple[int, ...], dict[int, float]]:
        """Convert node's partial selections to (selected_features, cat_selections)."""
        selected_features = []
        for g, nchoosek in enumerate(self.groups.nchooseks):
            for local_idx in node.partial_by_group[g]:
                selected_features.append(nchoosek.features[local_idx])
        selected_features_tuple = tuple(sorted(selected_features))

        cat_selections: dict[int, float] = {}
        n_nchoosek = len(self.groups.nchooseks)
        for i, cat_group in enumerate(self.groups.categoricals):
            g = n_nchoosek + i
            partial = node.partial_by_group[g]
            if partial:
                cat_selections[cat_group.dim] = cat_group.values[partial[0]]

        return selected_features_tuple, cat_selections

    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "size": len(self.value_cache),
        }

    # ─── Tree selection with Thompson Sampling ──────────────────────

    def _select_and_expand(self) -> tuple[TSNode, list[TSNode]]:
        """Select path through tree using TS and expand one new node."""
        node = self.root
        path = [node]

        while not node.is_terminal(self.groups):
            legal = self._legal_actions(node)
            limit = self._child_limit(node)
            unexpanded = [a for a in legal if a not in node.children]
            can_expand = len(node.children) < limit

            if can_expand and unexpanded:
                # Expand one new child
                action = self.rng.choice(unexpanded)
                child = self._apply_action(node, action)
                node.children[action] = child
                path.append(child)
                return child, path

            # Thompson Sampling selection among existing children
            if node.children:
                best_action = None
                best_score = float("-inf")
                for action, child in node.children.items():
                    score = self._ts_sample_score(child)
                    if score > best_score:
                        best_score = score
                        best_action = action

                node = node.children[best_action]
                path.append(node)
            else:
                break

        return node, path

    # ─── Rollout action selection ───────────────────────────────────

    def _ts_sample_rollout_action(
        self, group_idx: int, cardinality: int, legal_actions: list[int]
    ) -> int:
        """Sample rollout action using per-action TS posteriors."""
        best_action = legal_actions[0]
        best_score = float("-inf")

        for action in legal_actions:
            if self.rollout_mode == "ts_group_action":
                key = (group_idx, action)
            else:  # ts_group_card_action
                key = (group_idx, cardinality, action)

            stats = self.rollout_ts_stats.get(key, TSActionStats(0, 0.0, 0.0))
            score = self._ts_sample_action_score(stats)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    @staticmethod
    def _softmax_probs(logits: list[float]) -> list[float]:
        """Pure-math softmax (no torch dependency)."""
        max_logit = max(logits)
        exps = [math.exp(v - max_logit) for v in logits]
        total = sum(exps)
        return [e / total for e in exps]

    # ─── Softmax fallback methods (copied from production, no torch) ─

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to [0, 1] using running min-max."""
        reward_range = self.reward_max - self.reward_min
        if reward_range <= 0:
            return 0.5
        return (reward - self.reward_min) / reward_range

    def _score_rollout_actions(
        self, group_idx: int, legal_actions: list[int]
    ) -> dict[int, float]:
        """Score legal rollout actions using learned statistics."""
        scores: dict[int, float] = {}
        for action in legal_actions:
            key = (group_idx, action)
            stats = self.rollout_stats.get(key)
            if stats is not None and stats[0] > 0:
                visits, total_reward = stats
                mean_reward = total_reward / visits
                novelty = self.rollout_novelty_weight / math.sqrt(visits + 1)
                scores[action] = mean_reward + novelty
            else:
                scores[action] = self.rollout_novelty_weight
        return scores

    def _sample_softmax_rollout_action(
        self, group_idx: int, legal_actions: list[int]
    ) -> int:
        """Sample rollout action using softmax policy (pure-math, no torch)."""
        scores = self._score_rollout_actions(group_idx, legal_actions)
        n = len(legal_actions)

        logits = [scores[a] / self.rollout_tau for a in legal_actions]
        policy_probs = self._softmax_probs(logits)

        eps = self.rollout_epsilon
        probs = [(1.0 - eps) * p + eps / n for p in policy_probs]

        return self.rng.choices(legal_actions, weights=probs, k=1)[0]

    def _update_rollout_stats(
        self, trajectory: list[tuple[int, int, int]], reward: float
    ) -> None:
        """Update softmax rollout policy statistics from a completed trajectory."""
        for group_idx, _cardinality, action in trajectory:
            key = (group_idx, action)
            v, tot = self.rollout_stats.get(key, (0, 0.0))
            self.rollout_stats[key] = (v + 1, tot + reward)

    def _update_cardinality_stats(
        self, reward: float, selected_features: tuple[int, ...]
    ) -> None:
        """Update per-(group, cardinality) stats from a completed rollout."""
        selected_set = set(selected_features)
        for g, nchoosek in enumerate(self.groups.nchooseks):
            cardinality = sum(1 for f in nchoosek.features if f in selected_set)
            key = (g, cardinality)
            v, tot = self.cardinality_stats.get(key, (0, 0.0))
            self.cardinality_stats[key] = (v + 1, tot + reward)
            self.group_rollout_counts[g] += 1

    def _compute_adaptive_p_stop(
        self, group_idx: int, current_cardinality: int
    ) -> float:
        """Compute adaptive stop probability for softmax rollout mode."""
        if not self.adaptive_p_stop:
            return self.p_stop_rollout

        nchoosek = self.groups.nchooseks[group_idx]
        max_count = nchoosek.max_count

        stop_key = (group_idx, current_cardinality)
        stop_stats = self.cardinality_stats.get(stop_key)
        if stop_stats is None or stop_stats[0] == 0:
            return self.p_stop_rollout

        e_stop = stop_stats[1] / stop_stats[0]

        e_continue = float("-inf")
        has_continue_data = False
        for m in range(current_cardinality + 1, max_count + 1):
            cont_key = (group_idx, m)
            cont_stats = self.cardinality_stats.get(cont_key)
            if cont_stats is not None and cont_stats[0] > 0:
                mean_r = cont_stats[1] / cont_stats[0]
                if mean_r > e_continue:
                    e_continue = mean_r
                has_continue_data = True

        if not has_continue_data:
            return self.p_stop_rollout

        reward_range = self.reward_max - self.reward_min
        if reward_range <= 0:
            return self.p_stop_rollout

        tau = self.p_stop_temperature
        logit = (e_stop - e_continue) / (tau * reward_range)
        logit = max(-10.0, min(10.0, logit))
        p_learned = 1.0 / (1.0 + math.exp(-logit))

        group_visits = self.group_rollout_counts[group_idx]
        alpha = (
            min(1.0, group_visits / self.p_stop_warmup)
            if self.p_stop_warmup > 0
            else 1.0
        )
        return (1.0 - alpha) * self.p_stop_rollout + alpha * p_learned

    # ─── Rollout dispatch ───────────────────────────────────────────

    def _rollout(
        self, node: TSNode
    ) -> tuple[tuple[int, ...], dict[int, float], list[tuple[int, int, int]]]:
        """Rollout to terminal state with mode-dependent action selection.

        Returns:
            Tuple of (selected_features, cat_selections, trajectory) where
            trajectory is a list of (group_idx, cardinality, action) triples.
        """
        curr = TSNode(
            partial_by_group=tuple(node.partial_by_group),
            stopped_by_group=tuple(node.stopped_by_group),
            group_idx=node.group_idx,
        )
        trajectory: list[tuple[int, int, int]] = []

        while not curr.is_terminal(self.groups):
            legal = self._legal_actions(curr)
            if not legal:
                curr = TSNode(
                    partial_by_group=curr.partial_by_group,
                    stopped_by_group=curr.stopped_by_group,
                    group_idx=curr.group_idx + 1,
                )
                continue

            g = curr.group_idx
            cardinality = len(curr.partial_by_group[g])

            if self.rollout_mode == "uniform":
                action = self.rng.choice(legal)

            elif self.rollout_mode in ("ts_group_action", "ts_group_card_action"):
                action = self._ts_sample_rollout_action(g, cardinality, legal)

            elif self.rollout_mode == "softmax":
                # Softmax fallback: adaptive p_stop for NChooseK STOP decisions
                is_nchoosek = g < len(self.groups.nchooseks)
                if is_nchoosek and STOP in legal:
                    p_stop = self._compute_adaptive_p_stop(g, cardinality)
                    if self.rng.random() < p_stop:
                        trajectory.append((g, cardinality, STOP))
                        curr = self._apply_action(curr, STOP)
                        continue

                action = self._sample_softmax_rollout_action(g, legal)

            else:
                raise ValueError(f"Unknown rollout_mode: {self.rollout_mode}")

            trajectory.append((g, cardinality, action))
            curr = self._apply_action(curr, action)

        selected_features, cat_selections = self._get_selection(curr)
        return selected_features, cat_selections, trajectory

    # ─── Backpropagation ────────────────────────────────────────────

    def _backpropagate(self, path: list[TSNode], reward: float, is_novel: bool) -> None:
        """Backpropagate reward through path.

        Novel: update n_obs, sum_rewards, sum_sq_rewards, n_visits.
        Cache hit (no_update): only increment n_visits.
        Cache hit (variance_inflation): increment n_visits, decay n_obs
            to gradually widen posterior for exhausted subtrees.
        """
        if is_novel:
            for n in path:
                n.n_obs += 1
                n.sum_rewards += reward
                n.sum_sq_rewards += reward * reward
                n.n_visits += 1
        else:
            # Cache hit
            for n in path:
                n.n_visits += 1

                if self.cache_hit_mode == "variance_inflation":
                    # Decay n_obs to widen posterior, preserving mean
                    if n.n_obs > 1:
                        old_n = n.n_obs
                        new_n = max(1, int(old_n * self.variance_decay))
                        if new_n < old_n:
                            mean = n.sum_rewards / old_n
                            n.sum_rewards = mean * new_n
                            n.sum_sq_rewards *= new_n / old_n
                            n.n_obs = new_n

    def _update_rollout_ts_stats(
        self, trajectory: list[tuple[int, int, int]], reward: float
    ) -> None:
        """Update per-action TS stats from a completed rollout trajectory."""
        for group_idx, cardinality, action in trajectory:
            if self.rollout_mode == "ts_group_action":
                key = (group_idx, action)
            elif self.rollout_mode == "ts_group_card_action":
                key = (group_idx, cardinality, action)
            else:
                continue

            old = self.rollout_ts_stats.get(key, TSActionStats(0, 0.0, 0.0))
            self.rollout_ts_stats[key] = TSActionStats(
                n_obs=old.n_obs + 1,
                sum_rewards=old.sum_rewards + reward,
                sum_sq_rewards=old.sum_sq_rewards + reward * reward,
            )

    # ─── Main loop ──────────────────────────────────────────────────

    def run(self, n_iterations: int) -> tuple[tuple[int, ...], dict[int, float], float]:
        """Run MCTS-TS for specified number of iterations.

        Args:
            n_iterations: Number of MCTS iterations to run

        Returns:
            Tuple of (selected_features, cat_selections, best_value)
        """
        for _ in range(n_iterations):
            leaf, path = self._select_and_expand()

            if leaf.is_terminal(self.groups):
                selected_features, cat_selections = self._get_selection(leaf)
                trajectory: list[tuple[int, int, int]] = []
            else:
                # Rollout retry: re-roll on cache hits to discover novel selections
                selected_features, cat_selections, trajectory = self._rollout(leaf)
                for _attempt in range(self.max_rollout_retries):
                    key = self._make_cache_key(selected_features, cat_selections)
                    if key not in self.value_cache:
                        break
                    selected_features, cat_selections, trajectory = self._rollout(leaf)

            key = self._make_cache_key(selected_features, cat_selections)
            is_novel = key not in self.value_cache
            reward = self._cached_reward(selected_features, cat_selections)

            # Update reward range (used by softmax fallback)
            if reward < self.reward_min:
                self.reward_min = reward
            if reward > self.reward_max:
                self.reward_max = reward

            if reward > self.best_value:
                self.best_value = reward
                self.best_selection = (selected_features, cat_selections)

            # Update global stats for TS prior (mean and adaptive variance)
            if is_novel:
                self._novel_reward_sum += reward
                self._novel_reward_sq_sum += reward * reward
                self._novel_reward_count += 1

            # Backpropagate (raw reward, no normalization for TS tree)
            self._backpropagate(path, reward, is_novel)

            # Update rollout statistics
            if self.rollout_mode in ("ts_group_action", "ts_group_card_action"):
                self._update_rollout_ts_stats(trajectory, reward)
            elif self.rollout_mode == "softmax":
                self._update_rollout_stats(trajectory, reward)
                if self.adaptive_p_stop:
                    self._update_cardinality_stats(reward, selected_features)

        if self.best_selection is None:
            return (), {}, self.best_value
        return self.best_selection[0], self.best_selection[1], self.best_value
