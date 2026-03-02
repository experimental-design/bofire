"""MCTS-based acquisition function optimization for NChooseK and categorical constraints.

Uses Monte Carlo Tree Search with Normal-Inverse-Gamma (NIG) Thompson Sampling
to select which features are active (non-zero) and categorical values, then runs
BoTorch acquisition function optimization with inactive features fixed to zero
and categoricals fixed to selected values.

The NIG conjugate prior models rewards as drawn from Normal(mu, sigma^2) with
both mean and variance unknown. The marginal posterior for the mean is a
Student-t distribution with heavier tails at low observation counts, which
naturally handles the low-n regime without extra heuristics.

NIG prior: (mu, sigma^2) ~ NIG(mu0, n0, alpha0, beta0)
    mu0 = _global_mean() (running mean of novel rewards)
    n0 = pseudo-count (default 1.0, or adaptive from branching factor)
    alpha0 = nig_alpha0 parameter (default 1.0)
    beta0 = alpha0 * _prior_var() (so E[sigma^2] = prior_var)

After n observations with sufficient stats (n_obs, sum_rewards, sum_sq_rewards):
    x_bar = sum_rewards / n
    S = sum_sq_rewards - n * x_bar^2

    n0' = n0 + n
    mu0' = (n0 * mu0 + n * x_bar) / n0'
    alpha0' = alpha0 + n / 2
    beta0' = beta0 + S / 2 + (n0 * n * (x_bar - mu0)^2) / (2 * n0')

Marginal posterior for mu: Student-t with
    df = 2 * alpha0'
    location = mu0'
    scale = sqrt(beta0' / (alpha0' * n0'))
"""

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

import torch
from botorch.optim import optimize_acqf
from torch import Tensor


STOP = -1  # Sentinel for stopping selection in a group


class ActionStats(NamedTuple):
    """Sufficient statistics for an action.

    Args:
        n_obs: Number of observations
        sum_rewards: Sum of observed rewards
        sum_sq_rewards: Sum of squared observed rewards
    """

    n_obs: int
    sum_rewards: float
    sum_sq_rewards: float


class TrajectoryStep(NamedTuple):
    """One rollout step: which group and what action."""

    group: int
    action: int


class Selection(NamedTuple):
    """Terminal result: active features and categorical values."""

    features: tuple[int, ...]
    categoricals: dict[int, float]


# =============================================================================
# Group abstractions for MCTS
# =============================================================================


class Group(ABC):
    """Abstract base class for MCTS groups (NChooseK or Categorical)."""

    @property
    @abstractmethod
    def n_options(self) -> int:
        """Number of options/actions available in this group."""
        pass

    @abstractmethod
    def legal_actions(self, partial: tuple[int, ...], stopped: bool) -> list[int]:
        """Return legal actions given current partial selection."""
        pass

    @abstractmethod
    def is_complete(self, partial: tuple[int, ...], stopped: bool) -> bool:
        """Check if selection for this group is complete."""
        pass


@dataclass(frozen=True)
class NChooseK(Group):
    """NChooseK constraint specifying feature selection bounds.

    Args:
        features: Feature indices (can be non-contiguous, e.g., [0, 2, 4])
        min_count: Minimum number of features to select
        max_count: Maximum number of features to select
    """

    features: Sequence[int]
    min_count: int
    max_count: int

    def __post_init__(self):
        n = len(self.features)
        if not (0 <= self.min_count <= self.max_count <= n):
            raise ValueError(
                f"Invalid NChooseK constraint: require 0 <= min_count <= max_count <= n; "
                f"got min_count={self.min_count}, max_count={self.max_count}, n={n}"
            )

    @property
    def n_options(self) -> int:
        return len(self.features)

    @property
    def n_features(self) -> int:
        return len(self.features)

    def legal_actions(self, partial: tuple[int, ...], stopped: bool) -> list[int]:
        """Compute legal actions within this NChooseK group.

        Actions are indices into self.features (not the actual feature indices).
        Enforces strictly increasing selection order (combinations, not permutations).
        STOP is legal if len(partial) >= min_count and not already stopped.
        """
        n = self.n_features
        m = len(partial)

        if stopped or m >= self.max_count:
            return []

        actions: list[int] = []
        last = partial[-1] if partial else -1

        # Remaining picks needed after this action to satisfy min_count
        r_min_needed = max(0, self.min_count - (m + 1))
        # After picking index i, n - (i+1) items remain; require n - (i+1) >= r_min_needed
        end_inclusive = n - r_min_needed - 1
        start = last + 1

        if start <= end_inclusive:
            actions.extend(range(start, end_inclusive + 1))

        if m >= self.min_count:
            actions.append(STOP)

        return actions

    def is_complete(self, partial: tuple[int, ...], stopped: bool) -> bool:
        """NChooseK is complete when stopped or max_count reached."""
        return stopped or len(partial) >= self.max_count


@dataclass(frozen=True)
class Categorical(Group):
    """Categorical dimension with allowed values.

    Args:
        dim: The dimension index in the input space
        values: Sequence of allowed values for this dimension
    """

    dim: int
    values: Sequence[float]

    def __post_init__(self):
        if len(self.values) < 2:
            raise ValueError(
                f"CategoricalGroup requires at least two values, got {len(self.values)}"
            )

    @property
    def n_options(self) -> int:
        return len(self.values)

    def legal_actions(self, partial: tuple[int, ...], stopped: bool) -> list[int]:
        """Categorical must select exactly one value. No STOP action."""
        if len(partial) >= 1:
            # Already selected
            return []
        # All value indices are legal
        return list(range(self.n_options))

    def is_complete(self, partial: tuple[int, ...], stopped: bool) -> bool:
        """Categorical is complete when one value is selected."""
        return len(partial) >= 1


# =============================================================================
# Combined constraints container
# =============================================================================


@dataclass(frozen=True)
class Groups:
    """Collection of NChooseK constraints and categorical groups."""

    groups: list[Group]

    def __len__(self) -> int:
        return len(self.groups)

    @property
    def categoricals(self) -> list[Categorical]:
        return [g for g in self.groups if isinstance(g, Categorical)]

    @property
    def nchooseks(self) -> list[NChooseK]:
        return [g for g in self.groups if isinstance(g, NChooseK)]

    @property
    def all_nchoosek_features(self) -> list[int]:
        """All feature indices covered by NChooseK constraints."""
        all_feats = []
        for c in self.nchooseks:
            all_feats.extend(c.features)
        return all_feats

    @property
    def all_categorical_dims(self) -> list[int]:
        """All dimension indices that are categorical."""
        return [c.dim for c in self.categoricals]


# =============================================================================
# MCTS Node
# =============================================================================


@dataclass
class Node:
    """MCTS tree node with NIG sufficient statistics.

    Each node tracks Bayesian sufficient statistics (n_obs, sum_rewards,
    sum_sq_rewards) for the Normal-Inverse-Gamma posterior update, plus
    n_visits (which includes cache hits) for progressive widening.

    Args:
        partial_by_group: Partial selection per group (indices into group's options)
        stopped_by_group: Whether each group has stopped selecting (for NChooseK)
        group_idx: Current group being filled
        n_obs: Novel observation count (for NIG posterior updates)
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

    children: dict[int, "Node"] = field(default_factory=dict)

    def is_terminal(self, groups: Groups) -> bool:
        return self.group_idx >= len(groups)


# =============================================================================
# MCTS Implementation with NIG Thompson Sampling
# =============================================================================


class MCTS:
    """Monte Carlo Tree Search with Normal-Inverse-Gamma Thompson Sampling.

    Uses NIG conjugate posteriors for tree selection. The marginal posterior
    for the mean is a Student-t distribution with heavier tails at low
    observation counts, naturally preventing premature commitment.

    Args:
        groups: Collection of NChooseK and categorical constraints
        reward_fn: Function mapping (selected_features, categorical_selections) to reward
        nig_alpha0: NIG shape prior (default 1.0); lower = heavier tails at low n
        ts_prior_var: Prior variance (default 1.0); used to set beta0 = alpha0 * prior_var
        adaptive_prior_var: If True, use running empirical variance as prior variance
        cache_hit_mode: How to handle cache hits during backpropagation. Options:
            "no_update" - only increment n_visits (default virtual loss)
            "variance_inflation" - decay n_obs to widen posterior
            "pessimistic" - add pessimistic pseudo-observations
            "combined" - variance_inflation + pessimistic
            "adaptive_pessimistic" - pessimistic with exhaustion-scaled strength
            "adaptive_combined" - variance_inflation + adaptive_pessimistic
        variance_decay: Decay factor for variance inflation mode (default 0.95)
        rollout_mode: Rollout action selection policy. Options:
            "ts_group_action" - NIG Thompson Sampling per (group, action)
            "uniform" - fixed p_stop for NChooseK STOP, then uniform among non-STOP
        adaptive_n0: If True, set pseudo-count n0 = 1 + log(branching_factor)
        p_stop_rollout: Probability of early stop during uniform rollout (default 0.35)
        pw_k0: Progressive widening base constant (default 2.0)
        pw_alpha: Progressive widening exponent (default 0.6)
        max_rollout_retries: Maximum rollout retries on cache hit (default 3)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        groups: Groups,
        reward_fn: Callable[[tuple[int, ...], dict[int, float]], float],
        nig_alpha0: float = 1.0,
        ts_prior_var: float = 1.0,
        adaptive_prior_var: bool = True,
        cache_hit_mode: str = "variance_inflation",
        variance_decay: float = 0.95,
        rollout_mode: str = "ts_group_action",
        adaptive_n0: bool = False,
        p_stop_rollout: float = 0.35,
        pw_k0: float = 2.0,
        pw_alpha: float = 0.6,
        max_rollout_retries: int = 3,
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
        self.adaptive_n0 = adaptive_n0
        self.p_stop_rollout = p_stop_rollout
        self.pw_k0 = pw_k0
        self.pw_alpha = pw_alpha
        self.max_rollout_retries = max_rollout_retries
        self.rng = random.Random(seed)

        # Initialize root node
        n_groups = len(groups)
        self.root = Node(
            partial_by_group=tuple(() for _ in range(n_groups)),
            stopped_by_group=tuple(False for _ in range(n_groups)),
            group_idx=0,
        )

        # Best found so far
        self.best_selection: Optional[Selection] = None
        self.best_value: float = float("-inf")

        # Cache for terminal evaluations
        self.value_cache: dict[tuple, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Global novel reward tracking for NIG prior
        self._novel_reward_sum: float = 0.0
        self._novel_reward_sq_sum: float = 0.0
        self._novel_reward_count: int = 0

        # Rollout TS statistics: (group_idx, action) -> ActionStats
        self.rollout_ts_stats: dict[tuple[int, int], ActionStats] = {}

    # =========================================================================
    # NIG prior methods
    # =========================================================================

    def _global_mean(self) -> float:
        """Running mean of all novel rewards, used as the NIG prior center mu0."""
        if self._novel_reward_count == 0:
            return 0.0
        return self._novel_reward_sum / self._novel_reward_count

    def _prior_var(self) -> float:
        """Prior variance for the NIG model.

        When adaptive_prior_var is True and at least 2 novel rewards have been
        observed, returns the running empirical variance. Otherwise returns the
        fixed ts_prior_var.
        """
        if not self.adaptive_prior_var or self._novel_reward_count < 2:
            return self.ts_prior_var
        mean = self._global_mean()
        empirical_var = (
            self._novel_reward_sq_sum / self._novel_reward_count - mean * mean
        )
        return max(empirical_var, 1e-8)

    def _pessimistic_value(self) -> float:
        """Pessimistic pseudo-observation value: global_mean - global_std.

        Used by pessimistic and combined cache-hit modes to inject a
        below-average pseudo-observation that discourages over-visited branches.
        """
        mean = self._global_mean()
        if self._novel_reward_count < 2:
            return mean - math.sqrt(self.ts_prior_var)
        empirical_var = (
            self._novel_reward_sq_sum / self._novel_reward_count - mean * mean
        )
        return mean - math.sqrt(max(empirical_var, 1e-8))

    def _compute_n0(self, n_actions: int) -> float:
        """Compute pseudo-count n0 for the NIG prior.

        With adaptive_n0, n0 = 1 + log(branching_factor). Higher branching
        means each child is visited rarely early on, so more observations
        are needed before departing from the prior.
        """
        if not self.adaptive_n0:
            return 1.0
        return 1.0 + math.log(max(n_actions, 2))

    # =========================================================================
    # NIG Thompson Sampling
    # =========================================================================

    def _student_t_sample(self, df: float, loc: float, scale: float) -> float:
        """Sample from a Student-t distribution.

        Uses the representation: loc + scale * Z / sqrt(V / df)
        where Z ~ N(0,1) and V ~ chi-squared(df) = Gamma(df/2, 2).
        """
        z = self.rng.gauss(0, 1)
        v = self.rng.gammavariate(df / 2, 2)  # chi-squared(df)
        return loc + scale * z / math.sqrt(v / df)

    def _nig_sample(
        self,
        n_obs: int,
        sum_rewards: float,
        sum_sq_rewards: float,
        n0: float = 1.0,
    ) -> float:
        """Sample from the NIG posterior (marginal Student-t for the mean).

        Computes the Normal-Inverse-Gamma posterior update from sufficient
        statistics and draws a sample from the marginal Student-t distribution.

        The NIG prior is parameterized as:
            mu0 = _global_mean(), n0 = n0, alpha0 = nig_alpha0,
            beta0 = alpha0 * _prior_var()

        After n observations with sample mean x_bar and sum-of-squared-
        deviations S = sum(x_i^2) - n * x_bar^2, the posterior parameters are:
            n0' = n0 + n
            mu0' = (n0 * mu0 + n * x_bar) / n0'
            alpha0' = alpha0 + n/2
            beta0' = beta0 + S/2 + n0*n*(x_bar - mu0)^2 / (2*n0')

        The marginal posterior for mu is Student-t(df=2*alpha0', loc=mu0',
        scale=sqrt(beta0' / (alpha0' * n0'))).

        Args:
            n_obs: Number of novel observations
            sum_rewards: Sum of observed rewards
            sum_sq_rewards: Sum of squared observed rewards
            n0: Prior pseudo-count (default 1.0)

        Returns:
            A sample from the posterior predictive distribution for the mean
        """
        mu0 = self._global_mean()
        prior_var = self._prior_var()
        alpha0 = self.nig_alpha0
        beta0 = alpha0 * prior_var

        if n_obs == 0:
            # Prior: Student-t(df=2*alpha0, loc=mu0, scale=sqrt(beta0/(alpha0*n0)))
            df = 2 * alpha0
            scale = math.sqrt(beta0 / (alpha0 * n0))
            return self._student_t_sample(df, mu0, scale)

        x_bar = sum_rewards / n_obs
        s = sum_sq_rewards - n_obs * x_bar * x_bar
        s = max(s, 0.0)  # numerical safety

        # Posterior update
        n0_post = n0 + n_obs
        mu0_post = (n0 * mu0 + n_obs * x_bar) / n0_post
        alpha0_post = alpha0 + n_obs / 2
        beta0_post = beta0 + s / 2 + (n0 * n_obs * (x_bar - mu0) ** 2) / (2 * n0_post)

        df = 2 * alpha0_post
        scale = math.sqrt(beta0_post / (alpha0_post * n0_post))
        return self._student_t_sample(df, mu0_post, scale)

    # =========================================================================
    # Rollout TS methods
    # =========================================================================

    def _ts_sample_rollout_action(
        self, group_idx: int, legal_actions: list[int]
    ) -> int:
        """Sample rollout action using per-(group, action) NIG posteriors.

        For each legal action, draws a sample from the NIG posterior using the
        per-action sufficient statistics, then picks the action with the highest
        sample. STOP is scored like any other action.

        Args:
            group_idx: Index of the current group
            legal_actions: List of legal action indices (may include STOP)

        Returns:
            Selected action index
        """
        n0 = self._compute_n0(len(legal_actions))
        best_action = legal_actions[0]
        best_score = float("-inf")

        for action in legal_actions:
            key = (group_idx, action)
            stats = self.rollout_ts_stats.get(key, ActionStats(0, 0.0, 0.0))
            score = self._nig_sample(
                stats.n_obs, stats.sum_rewards, stats.sum_sq_rewards, n0
            )
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _update_rollout_ts_stats(
        self, trajectory: list[TrajectoryStep], reward: float
    ) -> None:
        """Update per-(group, action) NIG sufficient stats from a completed rollout.

        Args:
            trajectory: List of (group_idx, action) pairs from the rollout
            reward: Raw reward obtained from the terminal evaluation
        """
        for group_idx, action in trajectory:
            key = (group_idx, action)
            old = self.rollout_ts_stats.get(key, ActionStats(0, 0.0, 0.0))
            self.rollout_ts_stats[key] = ActionStats(
                n_obs=old.n_obs + 1,
                sum_rewards=old.sum_rewards + reward,
                sum_sq_rewards=old.sum_sq_rewards + reward * reward,
            )

    # =========================================================================
    # Tree infrastructure
    # =========================================================================

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

    def _child_limit(self, node: Node) -> int:
        """Progressive widening: max children based on visit count."""
        return max(1, int(self.pw_k0 * (max(1, node.n_visits) ** self.pw_alpha)))

    def _legal_actions(self, node: Node) -> list[int]:
        """Get legal actions for current group in node."""
        if node.is_terminal(self.groups):
            return []
        g = node.group_idx
        group = self.groups.groups[g]
        partial = node.partial_by_group[g]
        stopped = node.stopped_by_group[g]
        return group.legal_actions(partial, stopped)

    def _apply_action(self, node: Node, action: int) -> Node:
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
            # Check if group is complete
            if group.is_complete(partials[g], stoppeds[g]):
                next_g = g + 1
            else:
                next_g = g

        return Node(
            partial_by_group=tuple(partials),
            stopped_by_group=tuple(stoppeds),
            group_idx=next_g,
        )

    def _get_selection(self, node: Node) -> Selection:
        """Convert node's partial selections to (selected_features, cat_selections)."""
        # Extract NChooseK selections
        selected_features = []
        for g, nchoosek in enumerate(self.groups.nchooseks):
            for local_idx in node.partial_by_group[g]:
                selected_features.append(nchoosek.features[local_idx])
        selected_features_tuple = tuple(sorted(selected_features))

        # Extract categorical selections
        cat_selections: dict[int, float] = {}
        n_nchoosek = len(self.groups.nchooseks)
        for i, cat_group in enumerate(self.groups.categoricals):
            g = n_nchoosek + i
            partial = node.partial_by_group[g]
            if partial:
                cat_selections[cat_group.dim] = cat_group.values[partial[0]]

        return Selection(features=selected_features_tuple, categoricals=cat_selections)

    # =========================================================================
    # Tree selection with NIG Thompson Sampling
    # =========================================================================

    def _select_and_expand(self) -> tuple[Node, list[Node]]:
        """Select path through tree using NIG-TS and expand one new node.

        At each internal node, draws a Thompson sample from each child's NIG
        posterior and follows the child with the highest sample. When progressive
        widening allows expansion, expands a random unexplored action instead.
        """
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

            # NIG Thompson Sampling selection among existing children
            if node.children:
                n0 = self._compute_n0(len(node.children))
                best_action = None
                best_score = float("-inf")
                for action, child in node.children.items():
                    score = self._nig_sample(
                        child.n_obs, child.sum_rewards, child.sum_sq_rewards, n0
                    )
                    if score > best_score:
                        best_score = score
                        best_action = action

                node = node.children[best_action]
                path.append(node)
            else:
                break

        return node, path

    # =========================================================================
    # Rollout
    # =========================================================================

    def _rollout(
        self, node: Node
    ) -> tuple[tuple[int, ...], dict[int, float], list[TrajectoryStep]]:
        """Rollout to terminal state with mode-dependent action selection.

        Supports two rollout modes:
        - "ts_group_action": NIG Thompson Sampling per (group, action)
        - "uniform": adaptive p_stop for NChooseK STOP, then uniform among non-STOP

        Returns:
            Tuple of (selected_features, cat_selections, trajectory)
        """
        curr = Node(
            partial_by_group=tuple(node.partial_by_group),
            stopped_by_group=tuple(node.stopped_by_group),
            group_idx=node.group_idx,
        )
        trajectory: list[TrajectoryStep] = []

        while not curr.is_terminal(self.groups):
            legal = self._legal_actions(curr)
            if not legal:
                # No legal actions, advance group (group is complete)
                curr = Node(
                    partial_by_group=curr.partial_by_group,
                    stopped_by_group=curr.stopped_by_group,
                    group_idx=curr.group_idx + 1,
                )
                continue

            g = curr.group_idx

            if self.rollout_mode == "ts_group_action":
                # NIG Thompson Sampling: STOP scored like any other action
                action = self._ts_sample_rollout_action(g, legal)

            elif self.rollout_mode == "uniform":
                # Fixed p_stop for NChooseK STOP, then uniform
                is_nchoosek = g < len(self.groups.nchooseks)
                if is_nchoosek and STOP in legal:
                    if self.rng.random() < self.p_stop_rollout:
                        trajectory.append(TrajectoryStep(g, STOP))
                        curr = self._apply_action(curr, STOP)
                        continue

                # Choose uniformly among non-STOP actions
                choices = [a for a in legal if a != STOP]
                if not choices:
                    trajectory.append(TrajectoryStep(g, STOP))
                    curr = self._apply_action(curr, STOP)
                    continue

                action = self.rng.choice(choices)
            else:
                raise ValueError(f"Unknown rollout_mode: {self.rollout_mode}")

            trajectory.append(TrajectoryStep(g, action))
            curr = self._apply_action(curr, action)

        selected_features, cat_selections = self._get_selection(curr)
        return selected_features, cat_selections, trajectory

    # =========================================================================
    # Backpropagation with NIG cache-hit modes
    # =========================================================================

    def _backpropagate(self, path: list[Node], reward: float, is_novel: bool) -> None:
        """Backpropagate reward through path with NIG-aware cache-hit handling.

        For novel evaluations: updates n_obs, sum_rewards, sum_sq_rewards,
        and n_visits on each node in the path.

        For cache hits: always increments n_visits (for progressive widening),
        then applies the configured cache_hit_mode:
        - "no_update": only increment n_visits
        - "variance_inflation": decay n_obs to widen the NIG posterior
        - "pessimistic": add a pessimistic pseudo-observation
        - "combined": variance_inflation + pessimistic
        - "adaptive_pessimistic": pessimistic with exhaustion-scaled strength
        - "adaptive_combined": variance_inflation + adaptive_pessimistic

        Args:
            path: List of nodes from root to leaf
            reward: Raw reward value
            is_novel: Whether this is a novel (non-cached) evaluation
        """
        if is_novel:
            for n in path:
                n.n_obs += 1
                n.sum_rewards += reward
                n.sum_sq_rewards += reward * reward
                n.n_visits += 1
            return

        # Cache hit handling
        if self.cache_hit_mode in ("pessimistic", "combined"):
            pess = self._pessimistic_value()

        if self.cache_hit_mode in ("adaptive_pessimistic", "adaptive_combined"):
            g_mean = self._global_mean()
            if self._novel_reward_count < 2:
                g_std = math.sqrt(self.ts_prior_var)
            else:
                emp_var = (
                    self._novel_reward_sq_sum / self._novel_reward_count
                    - g_mean * g_mean
                )
                g_std = math.sqrt(max(emp_var, 1e-8))

        for n in path:
            n.n_visits += 1

            if self.cache_hit_mode == "no_update":
                pass

            elif self.cache_hit_mode == "variance_inflation":
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

            elif self.cache_hit_mode == "adaptive_pessimistic":
                novelty_rate = n.n_obs / max(1, n.n_visits)
                exhaustion = 1.0 - novelty_rate
                pess_value = g_mean - exhaustion * g_std
                n.n_obs += 1
                n.sum_rewards += pess_value
                n.sum_sq_rewards += pess_value * pess_value

            elif self.cache_hit_mode == "adaptive_combined":
                if n.n_obs > 1:
                    old_n = n.n_obs
                    new_n = max(1, int(old_n * self.variance_decay))
                    if new_n < old_n:
                        mean = n.sum_rewards / old_n
                        n.sum_rewards = mean * new_n
                        n.sum_sq_rewards *= new_n / old_n
                        n.n_obs = new_n
                novelty_rate = n.n_obs / max(1, n.n_visits)
                exhaustion = 1.0 - novelty_rate
                pess_value = g_mean - exhaustion * g_std
                n.n_obs += 1
                n.sum_rewards += pess_value
                n.sum_sq_rewards += pess_value * pess_value

    # =========================================================================
    # Main loop
    # =========================================================================

    def run(self, n_iterations: int) -> tuple[tuple[int, ...], dict[int, float], float]:
        """Run MCTS for specified number of iterations.

        Args:
            n_iterations: Number of MCTS iterations to run

        Returns:
            Tuple of (selected_features, cat_selections, best_value)
        """
        for _ in range(n_iterations):
            leaf, path = self._select_and_expand()

            if leaf.is_terminal(self.groups):
                selected_features, cat_selections = self._get_selection(leaf)
                trajectory: list[TrajectoryStep] = []
            else:
                # Rollout retry: if the rollout produces a cached terminal,
                # re-roll to try to discover a novel selection.
                selected_features, cat_selections, trajectory = self._rollout(leaf)
                for _attempt in range(self.max_rollout_retries):
                    key = self._make_cache_key(selected_features, cat_selections)
                    if key not in self.value_cache:
                        break
                    selected_features, cat_selections, trajectory = self._rollout(leaf)

            key = self._make_cache_key(selected_features, cat_selections)
            is_novel = key not in self.value_cache
            reward = self._cached_reward(selected_features, cat_selections)

            if reward > self.best_value:
                self.best_value = reward
                self.best_selection = Selection(
                    features=selected_features, categoricals=cat_selections
                )

            # Track novel reward statistics for NIG prior
            if is_novel:
                self._novel_reward_sum += reward
                self._novel_reward_sq_sum += reward * reward
                self._novel_reward_count += 1

            # Backpropagate with NIG cache-hit handling
            self._backpropagate(path, reward, is_novel)

            # Update rollout TS stats
            if self.rollout_mode == "ts_group_action":
                self._update_rollout_ts_stats(trajectory, reward)

        if self.best_selection is None:
            return (), {}, self.best_value
        return (
            self.best_selection.features,
            self.best_selection.categoricals,
            self.best_value,
        )

    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "size": len(self.value_cache),
        }


# =============================================================================
# Main optimization function
# =============================================================================


def optimize_acqf_mcts(
    acq_function,
    bounds: Tensor,
    nchooseks: list[tuple[list[int], int, int]] | None = None,
    cat_dims: Mapping[int, Sequence[float]] | None = None,
    # MCTS NIG parameters
    nig_alpha0: float = 1.0,
    ts_prior_var: float = 1.0,
    adaptive_prior_var: bool = True,
    cache_hit_mode: str = "variance_inflation",
    variance_decay: float = 0.95,
    rollout_mode: str = "ts_group_action",
    adaptive_n0: bool = False,
    p_stop_rollout: float = 0.35,
    num_iterations: int = 100,
    pw_k0: float = 2.0,
    pw_alpha: float = 0.6,
    max_rollout_retries: int = 3,
    # BoTorch acqf optimization parameters
    q: int = 1,
    raw_samples: int = 1024,
    num_restarts: int = 20,
    fixed_features: dict[int, float] | None = None,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    seed: int | None = None,
) -> tuple[Tensor, float]:
    """Optimize acquisition function with NChooseK and categorical constraints using MCTS.

    Uses MCTS with NIG Thompson Sampling to select which features are active
    (non-zero) and categorical values, then runs BoTorch optimization with
    inactive features fixed to zero and categoricals fixed to their selected values.

    Args:
        acq_function: BoTorch acquisition function to optimize
        bounds: 2 x d tensor of (lower, upper) bounds for each dimension
        nchooseks: List of NChooseK constraints as tuples of (features, min_count, max_count)
        cat_dims: Dictionary mapping categorical dimension indices to allowed values
        nig_alpha0: NIG shape prior (default 1.0)
        ts_prior_var: Prior variance for NIG model (default 1.0)
        adaptive_prior_var: Use running empirical variance as prior variance
        cache_hit_mode: Cache hit handling strategy
        variance_decay: Decay factor for variance inflation mode
        rollout_mode: Rollout action selection policy ("ts_group_action" or "uniform")
        adaptive_n0: Adapt pseudo-count from branching factor
        p_stop_rollout: Base probability of early stop during uniform rollout
        num_iterations: Number of MCTS iterations
        pw_k0: Progressive widening base constant
        pw_alpha: Progressive widening exponent
        max_rollout_retries: Maximum rollout retries on cache hit
        q: Batch size for acquisition function optimization
        raw_samples: Number of raw samples for initialization
        num_restarts: Number of optimization restarts
        fixed_features: Additional fixed features (combined with MCTS selections)
        inequality_constraints: Inequality constraints for BoTorch optimization
        equality_constraints: Equality constraints for BoTorch optimization
        seed: Random seed for reproducibility

    Returns:
        Tuple of (best_candidates, best_acq_value) where best_candidates is a
        q x d tensor of optimal points and best_acq_value is the acquisition value
    """
    d = bounds.shape[1]

    # Build NChooseK groups from tuples
    nchoosek_list = []
    if nchooseks:
        for features, min_count, max_count in nchooseks:
            nchoosek_list.append(
                NChooseK(features=features, min_count=min_count, max_count=max_count)
            )

    # Build categorical groups
    categorical_list = (
        [Categorical(dim=dim, values=list(values)) for dim, values in cat_dims.items()]
        if cat_dims
        else []
    )

    # Combine all groups
    all_groups = nchoosek_list + categorical_list
    groups = Groups(groups=all_groups)

    # All feature indices covered by NChooseK constraints
    nchoosek_features = set(groups.all_nchoosek_features)

    # Storage for best result across all MCTS evaluations
    best_candidates: Optional[Tensor] = None
    best_acq_value: float = float("-inf")

    def reward_fn(
        selected_features: tuple[int, ...], cat_selections: dict[int, float]
    ) -> float:
        nonlocal best_candidates, best_acq_value

        selected_set = set(selected_features)

        # Build fixed_features dict
        combined_fixed = {}

        # First add user-provided fixed features
        if fixed_features is not None:
            combined_fixed.update(fixed_features)

        # Fix inactive NChooseK features to 0
        inactive_features = nchoosek_features - selected_set
        for idx in inactive_features:
            combined_fixed[idx] = 0.0

        # Fix categorical dimensions to selected values
        for dim, value in cat_selections.items():
            combined_fixed[dim] = value

        candidates, acq_value = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=combined_fixed,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

        value = acq_value.item()

        # Track best
        if value > best_acq_value:
            best_acq_value = value
            best_candidates = candidates

        return value

    # Run MCTS
    mcts = MCTS(
        groups=groups,
        reward_fn=reward_fn,
        nig_alpha0=nig_alpha0,
        ts_prior_var=ts_prior_var,
        adaptive_prior_var=adaptive_prior_var,
        cache_hit_mode=cache_hit_mode,
        variance_decay=variance_decay,
        rollout_mode=rollout_mode,
        adaptive_n0=adaptive_n0,
        p_stop_rollout=p_stop_rollout,
        pw_k0=pw_k0,
        pw_alpha=pw_alpha,
        max_rollout_retries=max_rollout_retries,
        seed=seed,
    )

    mcts.run(n_iterations=num_iterations)

    # Handle case where no valid solution was found
    if best_candidates is None:
        # Return zeros with -inf value
        best_candidates = torch.zeros(q, d, dtype=bounds.dtype, device=bounds.device)
        best_acq_value = float("-inf")

    return best_candidates, best_acq_value
