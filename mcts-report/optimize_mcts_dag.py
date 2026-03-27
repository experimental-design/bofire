"""MCTS DAG with transposition table for NChooseK and categorical optimization.

Eliminates canonical ordering bias by:
1. Allowing all unselected features at every NChooseK node (not just those > last)
2. Using a transposition table to merge nodes with the same selected feature set
3. NIG-TS statistics merge cleanly across parent paths (no parent-dependent terms)

The tree becomes a DAG (directed acyclic graph) where nodes with identical
selected feature sets share statistics, regardless of selection order.
"""

import math
import random
from typing import Callable, Optional

from optimize_mcts_full import STOP, Groups, NChooseK
from optimize_mcts_ts import TSActionStats, TSNode


# =============================================================================
# Transposition key
# =============================================================================


def _transposition_key(
    group_idx: int,
    partial_by_group: tuple[tuple[int, ...], ...],
    stopped_by_group: tuple[bool, ...],
    groups: Groups,
) -> tuple:
    """Create canonical transposition key from DAG state.

    For NChooseK groups, uses frozenset of selected indices (order-independent).
    For Categorical groups, uses the tuple directly (at most 1 element).
    """
    canon = []
    for g_idx, group in enumerate(groups.groups):
        if isinstance(group, NChooseK):
            canon.append(frozenset(partial_by_group[g_idx]))
        else:
            canon.append(partial_by_group[g_idx])
    return (group_idx, tuple(canon), stopped_by_group)


# =============================================================================
# MCTS DAG with NIG Thompson Sampling
# =============================================================================


class MCTS_DAG:
    """MCTS with transposition table (DAG) and NIG Thompson Sampling.

    Removes canonical ordering constraint from NChooseK groups, allowing all
    unselected features at every decision point. A transposition table merges
    nodes with identical selected feature sets, preventing the exponential
    blowup that would otherwise result.

    Args:
        groups: Collection of NChooseK and categorical constraints
        reward_fn: Function mapping (selected_features, cat_selections) to reward
        nig_alpha0: NIG shape prior (default 1.0); lower = heavier tails at low n
        ts_prior_var: Prior variance (default 1.0); used to set beta0 = alpha0 * prior_var
        adaptive_prior_var: If True, set prior variance to running empirical variance
        cache_hit_mode: How to handle cache hits: "no_update", "variance_inflation",
            "pessimistic", or "combined"
        variance_decay: Decay factor for variance inflation mode (default 0.95)
        rollout_mode: Rollout policy: "uniform_subset", "ts_group_action", or "uniform"
        pw_k0: Progressive widening base constant (default 2.0)
        pw_alpha: Progressive widening exponent (default 0.6)
        max_rollout_retries: Maximum rollout retries on cache hit (default 3)
        adaptive_n0: If True, set n0 = 1 + log(branching_factor) to slow down
            premature convergence with the DAG's larger branching factors
        informed_expansion: If True, use rollout TS stats to prioritize which
            unexpanded actions to try first, instead of random selection
        separate_stop: If True, treat STOP as a binary decision (stop vs continue)
            before selecting which feature. This gives STOP a fair 50/50 comparison
            against the best feature alternative, fixing the STOP dilution problem
            where STOP competes with many features (1-out-of-N) in the flat action
            space. Critical for the DAG where branching stays constant.
        use_cache: If True (default), cache terminal evaluations. If False, every
            evaluation calls reward_fn fresh — required for stochastic/noisy reward
            functions (e.g. Sobol-based acqf sampling). With use_cache=False, every
            observation is novel, cache-hit modes are never triggered, and rollout
            retry is skipped.
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        groups: Groups,
        reward_fn: Callable[[tuple[int, ...], dict[int, float]], float],
        nig_alpha0: float = 1.0,
        ts_prior_var: float = 1.0,
        adaptive_prior_var: bool = False,
        cache_hit_mode: str = "no_update",
        variance_decay: float = 0.95,
        rollout_mode: str = "uniform",
        pw_k0: float = 2.0,
        pw_alpha: float = 0.6,
        max_rollout_retries: int = 3,
        adaptive_n0: bool = False,
        informed_expansion: bool = False,
        separate_stop: bool = False,
        use_cache: bool = True,
        seed: Optional[int] = None,
    ):
        self.groups = groups
        self.reward_fn = reward_fn
        self.nig_alpha0 = nig_alpha0
        self.ts_prior_var = ts_prior_var
        self.adaptive_prior_var = adaptive_prior_var
        self.cache_hit_mode = cache_hit_mode
        self.variance_decay = variance_decay
        self.rollout_mode = rollout_mode
        self.pw_k0 = pw_k0
        self.pw_alpha = pw_alpha
        self.max_rollout_retries = max_rollout_retries
        self.adaptive_n0 = adaptive_n0
        self.informed_expansion = informed_expansion
        self.separate_stop = separate_stop
        self.use_cache = use_cache
        self.rng = random.Random(seed)

        # Initialize root node
        n_groups = len(groups)
        self.root = TSNode(
            partial_by_group=tuple(() for _ in range(n_groups)),
            stopped_by_group=tuple(False for _ in range(n_groups)),
            group_idx=0,
        )

        # Transposition table: canonical key -> TSNode
        self.transposition_table: dict[tuple, TSNode] = {}
        root_key = _transposition_key(
            0, self.root.partial_by_group, self.root.stopped_by_group, self.groups
        )
        self.transposition_table[root_key] = self.root

        # Best found so far
        self.best_selection: Optional[tuple[tuple[int, ...], dict[int, float]]] = None
        self.best_value: float = float("-inf")

        # Cache for terminal evaluations
        self.value_cache: dict[tuple, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Global reward tracking for NIG prior center and adaptive variance
        self._novel_reward_sum: float = 0.0
        self._novel_reward_sq_sum: float = 0.0
        self._novel_reward_count: int = 0

        # Rollout TS statistics: (group_idx, action) -> TSActionStats
        self.rollout_ts_stats: dict[tuple, TSActionStats] = {}

    # --- NIG prior methods ----------------------------------------------------

    def _global_mean(self) -> float:
        """Running mean of all novel rewards (prior center mu0)."""
        if self._novel_reward_count == 0:
            return 0.0
        return self._novel_reward_sum / self._novel_reward_count

    def _prior_var(self) -> float:
        """Prior variance, either fixed or adaptive (empirical variance)."""
        if not self.adaptive_prior_var or self._novel_reward_count < 2:
            return self.ts_prior_var
        mean = self._global_mean()
        empirical_var = (
            self._novel_reward_sq_sum / self._novel_reward_count - mean * mean
        )
        return max(empirical_var, 1e-8)

    def _pessimistic_value(self) -> float:
        """Pessimistic pseudo-observation value: global_mean - global_std."""
        mean = self._global_mean()
        if self._novel_reward_count < 2:
            return mean - math.sqrt(self.ts_prior_var)
        empirical_var = (
            self._novel_reward_sq_sum / self._novel_reward_count - mean * mean
        )
        return mean - math.sqrt(max(empirical_var, 1e-8))

    def _compute_n0(self, n_actions: int) -> float:
        """Compute pseudo-count n0 from branching factor.

        With adaptive_n0, n0 = 1 + log(branching_factor). Higher branching
        means each child is visited rarely early on, so we need more
        observations before departing from the prior. This prevents premature
        lock-in with the DAG's larger action spaces.
        """
        if not self.adaptive_n0:
            return 1.0
        return 1.0 + math.log(max(n_actions, 2))

    # --- NIG Student-t sampling -----------------------------------------------

    def _student_t_sample(self, df: float, loc: float, scale: float) -> float:
        """Sample from a Student-t distribution.

        Uses the representation: loc + scale * Z / sqrt(V / df)
        where Z ~ N(0,1) and V ~ chi-squared(df) = Gamma(df/2, 2).
        """
        z = self.rng.gauss(0, 1)
        v = self.rng.gammavariate(df / 2, 2)
        return loc + scale * z / math.sqrt(v / df)

    def _nig_sample_score(self, node: TSNode, n0: float = 1.0) -> float:
        """Sample from node's NIG posterior (marginal Student-t) for tree selection."""
        mu0 = self._global_mean()
        prior_var = self._prior_var()
        alpha0 = self.nig_alpha0
        beta0 = alpha0 * prior_var

        n = node.n_obs
        if n == 0:
            df = 2 * alpha0
            scale = math.sqrt(beta0 / (alpha0 * n0))
            return self._student_t_sample(df, mu0, scale)

        x_bar = node.sum_rewards / n
        s = node.sum_sq_rewards - n * x_bar * x_bar
        s = max(s, 0.0)

        n0_post = n0 + n
        mu0_post = (n0 * mu0 + n * x_bar) / n0_post
        alpha0_post = alpha0 + n / 2
        beta0_post = beta0 + s / 2 + (n0 * n * (x_bar - mu0) ** 2) / (2 * n0_post)

        df = 2 * alpha0_post
        scale = math.sqrt(beta0_post / (alpha0_post * n0_post))
        return self._student_t_sample(df, mu0_post, scale)

    def _nig_sample_action_score(self, stats: TSActionStats, n0: float = 1.0) -> float:
        """Sample from a TSActionStats NIG posterior (for rollout actions)."""
        mu0 = self._global_mean()
        prior_var = self._prior_var()
        alpha0 = self.nig_alpha0
        beta0 = alpha0 * prior_var

        n = stats.n_obs
        if n == 0:
            df = 2 * alpha0
            scale = math.sqrt(beta0 / (alpha0 * n0))
            return self._student_t_sample(df, mu0, scale)

        x_bar = stats.sum_rewards / n
        s = stats.sum_sq_rewards - n * x_bar * x_bar
        s = max(s, 0.0)

        n0_post = n0 + n
        mu0_post = (n0 * mu0 + n * x_bar) / n0_post
        alpha0_post = alpha0 + n / 2
        beta0_post = beta0 + s / 2 + (n0 * n * (x_bar - mu0) ** 2) / (2 * n0_post)

        df = 2 * alpha0_post
        scale = math.sqrt(beta0_post / (alpha0_post * n0_post))
        return self._student_t_sample(df, mu0_post, scale)

    # --- DAG-specific legal actions -------------------------------------------

    def _nchoosek_legal_actions_dag(
        self, group: NChooseK, partial: tuple[int, ...], stopped: bool
    ) -> list[int]:
        """Legal actions for NChooseK without canonical ordering.

        Returns ALL unselected features (not just those > last), plus STOP
        when min_count is satisfied. This is THE CORE CHANGE from tree MCTS.
        """
        if stopped or len(partial) >= group.max_count:
            return []

        selected = set(partial)
        actions = [i for i in range(group.n_features) if i not in selected]

        if len(partial) >= group.min_count:
            actions.append(STOP)

        return actions

    def _legal_actions(self, node: TSNode) -> list[int]:
        """Get legal actions for current group in node."""
        if node.is_terminal(self.groups):
            return []
        g = node.group_idx
        group = self.groups.groups[g]
        partial = node.partial_by_group[g]
        stopped = node.stopped_by_group[g]

        if isinstance(group, NChooseK):
            return self._nchoosek_legal_actions_dag(group, partial, stopped)
        else:
            return group.legal_actions(partial, stopped)

    # --- DAG-aware apply_action -----------------------------------------------

    def _apply_action(self, node: TSNode, action: int) -> TSNode:
        """Apply action, returning shared node from transposition table if exists."""
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

        new_partial = tuple(partials)
        new_stopped = tuple(stoppeds)

        # Lookup in transposition table
        key = _transposition_key(next_g, new_partial, new_stopped, self.groups)
        if key in self.transposition_table:
            return self.transposition_table[key]

        # Create new node and register
        new_node = TSNode(
            partial_by_group=new_partial,
            stopped_by_group=new_stopped,
            group_idx=next_g,
        )
        self.transposition_table[key] = new_node
        return new_node

    # --- Cache / utility methods ----------------------------------------------

    def _make_cache_key(
        self, selected_features: tuple[int, ...], cat_selections: dict[int, float]
    ) -> tuple:
        """Create hashable cache key from selection."""
        return (selected_features, frozenset(cat_selections.items()))

    def _cached_reward(
        self, selected_features: tuple[int, ...], cat_selections: dict[int, float]
    ) -> float:
        """Get cached reward or compute and cache it.

        With use_cache=False, always calls reward_fn fresh (for stochastic rewards).
        """
        if not self.use_cache:
            self.cache_misses += 1
            return self.reward_fn(selected_features, cat_selections)
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

    def _get_selection_from_state(
        self,
        partial_by_group: list[list[int]],
        stopped_by_group: list[bool],
    ) -> tuple[tuple[int, ...], dict[int, float]]:
        """Convert mutable rollout state to (selected_features, cat_selections)."""
        selected_features = []
        for g, nchoosek in enumerate(self.groups.nchooseks):
            for local_idx in partial_by_group[g]:
                selected_features.append(nchoosek.features[local_idx])
        selected_features_tuple = tuple(sorted(selected_features))

        cat_selections: dict[int, float] = {}
        n_nchoosek = len(self.groups.nchooseks)
        for i, cat_group in enumerate(self.groups.categoricals):
            g = n_nchoosek + i
            partial = partial_by_group[g]
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

    # --- Tree selection with NIG Thompson Sampling ----------------------------

    def _pick_expansion_action(self, node: TSNode, unexpanded: list[int]) -> int:
        """Pick which unexpanded action to expand next.

        With informed_expansion, uses rollout TS stats to sample from
        NIG posteriors for each candidate and picks the highest-scoring one.
        Otherwise picks uniformly at random.
        """
        if not self.informed_expansion or not self.rollout_ts_stats:
            return self.rng.choice(unexpanded)

        g = node.group_idx
        best_action = unexpanded[0]
        best_score = float("-inf")
        for action in unexpanded:
            key = (g, action)
            stats = self.rollout_ts_stats.get(key, TSActionStats(0, 0.0, 0.0))
            score = self._nig_sample_action_score(stats)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _select_action_separate_stop(self, node: TSNode, n0: float) -> int:
        """Select among children using binary STOP-vs-best-feature comparison.

        When STOP is among the children, first sample STOP's NIG score, then
        sample the best feature's NIG score, and compare. This gives STOP a
        fair 50/50 chance instead of being diluted among many feature actions.
        Falls back to normal NIG-TS when STOP is not among children.
        """
        if STOP not in node.children:
            # No STOP — normal NIG-TS among all children
            best_action = None
            best_score = float("-inf")
            for action, child in node.children.items():
                score = self._nig_sample_score(child, n0=n0)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action

        # Binary comparison: STOP vs best feature
        stop_score = self._nig_sample_score(node.children[STOP], n0=n0)

        feature_children = {a: c for a, c in node.children.items() if a != STOP}
        if not feature_children:
            return STOP

        best_feat_action = None
        best_feat_score = float("-inf")
        for action, child in feature_children.items():
            score = self._nig_sample_score(child, n0=n0)
            if score > best_feat_score:
                best_feat_score = score
                best_feat_action = action

        if stop_score >= best_feat_score:
            return STOP
        return best_feat_action

    def _select_and_expand(self) -> tuple[TSNode, list[TSNode]]:
        """Select path through DAG using NIG-TS and expand one new node."""
        node = self.root
        path = [node]

        while not node.is_terminal(self.groups):
            legal = self._legal_actions(node)
            limit = self._child_limit(node)
            unexpanded = [a for a in legal if a not in node.children]
            can_expand = len(node.children) < limit

            if can_expand and unexpanded:
                # Expand: prioritize STOP first when legal and unexpanded
                if self.separate_stop and STOP in unexpanded:
                    action = STOP
                else:
                    action = self._pick_expansion_action(node, unexpanded)
                child = self._apply_action(node, action)
                node.children[action] = child
                path.append(child)
                return child, path

            # NIG Thompson Sampling selection among existing children
            if node.children:
                n0 = self._compute_n0(len(node.children))

                if self.separate_stop:
                    best_action = self._select_action_separate_stop(node, n0)
                else:
                    best_action = None
                    best_score = float("-inf")
                    for action, child in node.children.items():
                        score = self._nig_sample_score(child, n0=n0)
                        if score > best_score:
                            best_score = score
                            best_action = action

                node = node.children[best_action]
                path.append(node)
            else:
                break

        return node, path

    # --- Rollout action selection ---------------------------------------------

    def _ts_sample_rollout_action(
        self, group_idx: int, legal_actions: list[int]
    ) -> int:
        """Sample rollout action using per-(group, action) NIG posteriors.

        With separate_stop, uses binary STOP-vs-best-feature comparison
        to give STOP fair representation in rollout decisions.
        """
        if self.separate_stop and STOP in legal_actions:
            # Binary: STOP vs best feature
            stop_key = (group_idx, STOP)
            stop_stats = self.rollout_ts_stats.get(stop_key, TSActionStats(0, 0.0, 0.0))
            stop_score = self._nig_sample_action_score(stop_stats)

            feature_actions = [a for a in legal_actions if a != STOP]
            if not feature_actions:
                return STOP

            best_feat = feature_actions[0]
            best_feat_score = float("-inf")
            for action in feature_actions:
                key = (group_idx, action)
                stats = self.rollout_ts_stats.get(key, TSActionStats(0, 0.0, 0.0))
                score = self._nig_sample_action_score(stats)
                if score > best_feat_score:
                    best_feat_score = score
                    best_feat = action

            if stop_score >= best_feat_score:
                return STOP
            return best_feat

        # Normal: NIG-TS among all actions
        best_action = legal_actions[0]
        best_score = float("-inf")

        for action in legal_actions:
            key = (group_idx, action)
            stats = self.rollout_ts_stats.get(key, TSActionStats(0, 0.0, 0.0))
            score = self._nig_sample_action_score(stats)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # --- Rollout dispatch -----------------------------------------------------

    def _rollout(
        self, node: TSNode
    ) -> tuple[tuple[int, ...], dict[int, float], list[tuple[int, int]]]:
        """Rollout to terminal state with mode-dependent action selection.

        Operates on mutable state, NOT the transposition table, to avoid
        bloating the table with rollout-only nodes.

        Returns:
            Tuple of (selected_features, cat_selections, trajectory) where
            trajectory is a list of (group_idx, action) tuples.
        """
        # Work with mutable copies
        partial_by_group = [list(p) for p in node.partial_by_group]
        stopped_by_group = list(node.stopped_by_group)
        current_g = node.group_idx
        trajectory: list[tuple[int, int]] = []

        n_groups = len(self.groups)

        if self.rollout_mode == "uniform_subset":
            # Fast rollout: for each incomplete group, directly sample a subset
            while current_g < n_groups:
                group = self.groups.groups[current_g]

                if isinstance(group, NChooseK):
                    if (
                        not stopped_by_group[current_g]
                        and len(partial_by_group[current_g]) < group.max_count
                    ):
                        selected = set(partial_by_group[current_g])
                        available = [
                            i for i in range(group.n_features) if i not in selected
                        ]

                        m = len(partial_by_group[current_g])
                        min_more = max(0, group.min_count - m)
                        max_more = min(len(available), group.max_count - m)

                        if max_more > 0 and min_more <= max_more:
                            k = self.rng.randint(min_more, max_more)
                            chosen = self.rng.sample(available, k)
                            for feat in chosen:
                                trajectory.append((current_g, feat))
                                partial_by_group[current_g].append(feat)

                    stopped_by_group[current_g] = True
                    current_g += 1
                else:
                    # Categorical: pick uniformly
                    if not partial_by_group[current_g]:
                        action = self.rng.randrange(group.n_options)
                        trajectory.append((current_g, action))
                        partial_by_group[current_g].append(action)
                    current_g += 1

        else:
            # Step-by-step rollout (uniform or ts_group_action)
            while current_g < n_groups:
                group = self.groups.groups[current_g]

                if isinstance(group, NChooseK):
                    partial_tuple = tuple(partial_by_group[current_g])
                    stopped = stopped_by_group[current_g]
                    legal = self._nchoosek_legal_actions_dag(
                        group, partial_tuple, stopped
                    )
                else:
                    partial_tuple = tuple(partial_by_group[current_g])
                    stopped = stopped_by_group[current_g]
                    legal = group.legal_actions(partial_tuple, stopped)

                if not legal:
                    current_g += 1
                    continue

                if self.rollout_mode == "uniform":
                    if self.separate_stop and STOP in legal:
                        # 50/50 stop vs continue
                        if self.rng.random() < 0.5:
                            action = STOP
                        else:
                            features = [a for a in legal if a != STOP]
                            action = self.rng.choice(features) if features else STOP
                    else:
                        action = self.rng.choice(legal)
                elif self.rollout_mode == "ts_group_action":
                    action = self._ts_sample_rollout_action(current_g, legal)
                else:
                    raise ValueError(f"Unknown rollout_mode: {self.rollout_mode}")

                trajectory.append((current_g, action))

                # Apply action to mutable state
                if action == STOP:
                    stopped_by_group[current_g] = True
                    current_g += 1
                else:
                    partial_by_group[current_g].append(action)
                    if group.is_complete(
                        tuple(partial_by_group[current_g]),
                        stopped_by_group[current_g],
                    ):
                        current_g += 1

        selected_features, cat_selections = self._get_selection_from_state(
            partial_by_group, stopped_by_group
        )
        return selected_features, cat_selections, trajectory

    # --- Backpropagation ------------------------------------------------------

    def _backpropagate(self, path: list[TSNode], reward: float, is_novel: bool) -> None:
        """Backpropagate reward through path.

        Novel: update n_obs, sum_rewards, sum_sq_rewards, n_visits.
        Cache hit handling depends on cache_hit_mode:
        - no_update: only increment n_visits
        - variance_inflation: decay n_obs to widen posterior
        - pessimistic: inject pessimistic pseudo-observation
        - combined: variance inflation + pessimistic
        """
        if is_novel:
            for n in path:
                n.n_obs += 1
                n.sum_rewards += reward
                n.sum_sq_rewards += reward * reward
                n.n_visits += 1
        else:
            if self.cache_hit_mode in ("pessimistic", "combined"):
                pess = self._pessimistic_value()

            for n in path:
                n.n_visits += 1

                if self.cache_hit_mode == "variance_inflation":
                    if n.n_obs > 1:
                        old_n = n.n_obs
                        new_n = max(1, int(old_n * self.variance_decay))
                        if new_n < old_n:
                            mean = n.sum_rewards / old_n
                            n.sum_rewards = mean * new_n
                            n.sum_sq_rewards *= new_n / old_n
                            n.n_obs = new_n
                elif self.cache_hit_mode == "pessimistic":
                    n.n_obs += 1
                    n.sum_rewards += pess
                    n.sum_sq_rewards += pess * pess
                elif self.cache_hit_mode == "combined":
                    if n.n_obs > 1:
                        old_n = n.n_obs
                        new_n = max(1, int(old_n * self.variance_decay))
                        if new_n < old_n:
                            mean = n.sum_rewards / old_n
                            n.sum_rewards = mean * new_n
                            n.sum_sq_rewards *= new_n / old_n
                            n.n_obs = new_n
                    n.n_obs += 1
                    n.sum_rewards += pess
                    n.sum_sq_rewards += pess * pess

    def _update_rollout_ts_stats(
        self, trajectory: list[tuple[int, int]], reward: float
    ) -> None:
        """Update per-(group, action) TS stats from rollout trajectory."""
        if self.rollout_mode != "ts_group_action":
            return

        for group_idx, action in trajectory:
            key = (group_idx, action)
            old = self.rollout_ts_stats.get(key, TSActionStats(0, 0.0, 0.0))
            self.rollout_ts_stats[key] = TSActionStats(
                n_obs=old.n_obs + 1,
                sum_rewards=old.sum_rewards + reward,
                sum_sq_rewards=old.sum_sq_rewards + reward * reward,
            )

    # --- Main loop ------------------------------------------------------------

    def run(self, n_iterations: int) -> tuple[tuple[int, ...], dict[int, float], float]:
        """Run MCTS-DAG for specified number of iterations.

        Args:
            n_iterations: Number of MCTS iterations to run

        Returns:
            Tuple of (selected_features, cat_selections, best_value)
        """
        for _ in range(n_iterations):
            leaf, path = self._select_and_expand()

            if leaf.is_terminal(self.groups):
                selected_features, cat_selections = self._get_selection(leaf)
                trajectory: list[tuple[int, int]] = []
            else:
                selected_features, cat_selections, trajectory = self._rollout(leaf)
                # Rollout retry: re-roll on cache hits to discover novel selections
                # Skip when use_cache=False (every eval is fresh)
                if self.use_cache:
                    for _attempt in range(self.max_rollout_retries):
                        key = self._make_cache_key(selected_features, cat_selections)
                        if key not in self.value_cache:
                            break
                        selected_features, cat_selections, trajectory = self._rollout(
                            leaf
                        )

            if self.use_cache:
                key = self._make_cache_key(selected_features, cat_selections)
                is_novel = key not in self.value_cache
            else:
                is_novel = True
            reward = self._cached_reward(selected_features, cat_selections)

            if reward > self.best_value:
                self.best_value = reward
                self.best_selection = (selected_features, cat_selections)

            # Update global stats for NIG prior (mean and adaptive variance)
            if is_novel:
                self._novel_reward_sum += reward
                self._novel_reward_sq_sum += reward * reward
                self._novel_reward_count += 1

            # Backpropagate (raw reward, no normalization for NIG-TS)
            self._backpropagate(path, reward, is_novel)

            # Update rollout statistics
            self._update_rollout_ts_stats(trajectory, reward)

        if self.best_selection is None:
            return (), {}, self.best_value
        return self.best_selection[0], self.best_selection[1], self.best_value
