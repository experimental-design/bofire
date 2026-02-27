"""MCTS-based acquisition function optimization for NChooseK and categorical constraints.

Uses Monte Carlo Tree Search to select which features are active (non-zero) and
categorical values, then runs BoTorch acquisition function optimization with
inactive features fixed to zero and categoricals fixed to selected values.
"""

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from botorch.optim import optimize_acqf
from torch import Tensor


STOP = -1  # Sentinel for stopping selection in a group


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
    """MCTS tree node.

    Args:
        partial_by_group: Partial selection per group (indices into group's options)
        stopped_by_group: Whether each group has stopped selecting (for NChooseK)
        group_idx: Current group being filled
        n_visits: Visit count for this node
        w_total: Total accumulated reward
        children: Child nodes keyed by action (int index or STOP)
    """

    partial_by_group: tuple[tuple[int, ...], ...]
    stopped_by_group: tuple[bool, ...]
    group_idx: int

    n_visits: int = 0
    w_total: float = 0.0

    children: dict[int, "Node"] = field(default_factory=dict)

    def is_terminal(self, groups: Groups) -> bool:
        return self.group_idx >= len(groups)

    def mean_value(self) -> float:
        return self.w_total / self.n_visits if self.n_visits > 0 else 0.0


# =============================================================================
# MCTS Implementation
# =============================================================================


class MCTS:
    """Monte Carlo Tree Search for NChooseK and categorical optimization.

    Uses UCT selection, RAVE action value estimation, and progressive widening.
    Selects which features are active and categorical values via tree search,
    evaluating terminals with a provided reward function.

    Args:
        constraints: Collection of NChooseK and categorical constraints
        reward_fn: Function mapping (selected_features, categorical_selections) to reward
        c_uct: UCT exploration constant (default 0.01)
        k_rave: RAVE blending decay parameter (default 300.0)
        p_stop_rollout: Probability of early stop during rollout (default 0.35)
        pw_k0: Progressive widening base constant (default 2.0)
        pw_alpha: Progressive widening exponent (default 0.6)
        max_rollout_retries: Maximum rollout retries on cache hit (default 3)
        adaptive_p_stop: Enable adaptive per-group stop probability (default True)
        p_stop_warmup: Per-group rollout count before full blending (default 20)
        p_stop_temperature: Sigmoid sharpness for adaptive p_stop (default 0.25)
        normalize_rewards: Normalize rewards to [0, 1] before backpropagation (default True)
        rollout_policy: Enable learned softmax rollout policy (default False)
        rollout_epsilon: Epsilon-mix weight for uniform exploration (default 0.3)
        rollout_tau: Softmax temperature for rollout policy (default 1.0)
        rollout_novelty_weight: Novelty bonus coefficient beta/sqrt(n+1) (default 1.0)
        context_rave: Use context-aware RAVE instead of global RAVE (default False)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        groups: Groups,
        reward_fn: Callable[[tuple[int, ...], dict[int, float]], float],
        c_uct: float = 0.01,
        k_rave: float = 300.0,
        p_stop_rollout: float = 0.35,
        pw_k0: float = 2.0,
        pw_alpha: float = 0.6,
        max_rollout_retries: int = 3,
        adaptive_p_stop: bool = True,
        p_stop_warmup: int = 20,
        p_stop_temperature: float = 0.25,
        normalize_rewards: bool = True,
        rollout_policy: bool = True,
        rollout_epsilon: float = 0.3,
        rollout_tau: float = 1.0,
        rollout_novelty_weight: float = 1.0,
        context_rave: bool = False,
        seed: Optional[int] = None,
    ):
        self.groups = groups
        self.reward_fn = reward_fn
        self.c_uct = c_uct
        self.k_rave = k_rave
        self.p_stop_rollout = p_stop_rollout
        self.pw_k0 = pw_k0
        self.pw_alpha = pw_alpha
        self.max_rollout_retries = max_rollout_retries
        self.adaptive_p_stop = adaptive_p_stop
        self.p_stop_warmup = p_stop_warmup
        self.p_stop_temperature = p_stop_temperature
        self.normalize_rewards = normalize_rewards
        self.rollout_policy = rollout_policy
        self.rollout_epsilon = rollout_epsilon
        self.rollout_tau = rollout_tau
        self.rollout_novelty_weight = rollout_novelty_weight
        self.context_rave = context_rave
        self.rng = random.Random(seed)

        # Initialize root node
        n_groups = len(groups)
        self.root = Node(
            partial_by_group=tuple(() for _ in range(n_groups)),
            stopped_by_group=tuple(False for _ in range(n_groups)),
            group_idx=0,
        )

        # Best found so far: (selected_features, categorical_selections)
        # Example: ((0, 2, 5), {3: 1.0, 4: 0.0}) means features 0, 2, 5 are
        # active (from NChooseK groups), dim 3 has categorical value 1.0, and
        # dim 4 has categorical value 0.0.
        self.best_selection: Optional[tuple[tuple[int, ...], dict[int, float]]] = None
        self.best_value: float = float("-inf")

        # Cache for terminal evaluations
        # Key: (selected_features_tuple, frozenset of categorical items)
        self.value_cache: dict[tuple, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # RAVE statistics: global_id -> (visits, total_reward)
        self.global_offsets = self._compute_group_offsets()
        self.rave_stats: dict[int, tuple[int, float]] = {}

        # Adaptive p_stop statistics: (group_idx, cardinality) -> (visits, total_reward)
        self.cardinality_stats: dict[tuple[int, int], tuple[int, float]] = {}
        n_nchoosek = len(self.groups.nchooseks)
        self.group_rollout_counts: list[int] = [0] * n_nchoosek
        self.reward_min: float = float("inf")
        self.reward_max: float = float("-inf")

        # Rollout policy statistics: (group_idx, action) -> (visits, total_reward)
        self.rollout_stats: dict[tuple[int, int], tuple[int, float]] = {}

        # Context-aware RAVE: (group_idx, cardinality, action) -> (visits, total_reward)
        self.context_rave_stats: dict[tuple[int, int, int], tuple[int, float]] = {}

    def _compute_group_offsets(self) -> list[int]:
        """Compute offset for each group to create global action IDs."""
        offsets = []
        acc = 0
        for group in self.groups.groups:
            offsets.append(acc)
            acc += group.n_options
        return offsets

    def _global_action_id(self, group_idx: int, local_idx: int) -> int:
        """Convert (group, local_index) to global action ID for RAVE."""
        return self.global_offsets[group_idx] + local_idx

    def _update_cardinality_stats(
        self, reward: float, selected_features: tuple[int, ...]
    ) -> None:
        """Update per-(group, cardinality) stats from a completed rollout.

        Reverse-maps selected_features to per-group cardinalities and updates
        the cardinality_stats dict.
        """
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
        """Compute adaptive stop probability for a group at a given cardinality.

        Uses learned cardinality statistics to decide whether stopping is better
        than continuing. Returns fixed p_stop_rollout when disabled, no data is
        available, or reward range is zero. During warmup, the learned probability
        is linearly blended with p_stop_rollout:
            p = (1 - alpha) * p_stop_rollout + alpha * p_learned
        where alpha = min(1, group_visits / p_stop_warmup), so the learned signal
        gradually replaces the fixed prior as more data is collected.

        Args:
            group_idx: Index of the NChooseK group
            current_cardinality: Number of features already selected in this group

        Returns:
            Stop probability in [0, 1]
        """
        if not self.adaptive_p_stop:
            return self.p_stop_rollout

        nchoosek = self.groups.nchooseks[group_idx]
        max_count = nchoosek.max_count

        # No data for stopping at this cardinality
        stop_key = (group_idx, current_cardinality)
        stop_stats = self.cardinality_stats.get(stop_key)
        if stop_stats is None or stop_stats[0] == 0:
            return self.p_stop_rollout

        # E_stop: mean reward when this group stopped at current_cardinality
        e_stop = stop_stats[1] / stop_stats[0]

        # E_continue: max mean reward among higher cardinalities
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

        # Reward range for normalization
        reward_range = self.reward_max - self.reward_min
        if reward_range <= 0:
            return self.p_stop_rollout

        # Sigmoid on normalized difference
        tau = self.p_stop_temperature
        logit = (e_stop - e_continue) / (tau * reward_range)
        logit = max(-10.0, min(10.0, logit))  # clamp
        p_learned = 1.0 / (1.0 + math.exp(-logit))

        # Warmup blending
        group_visits = self.group_rollout_counts[group_idx]
        alpha = (
            min(1.0, group_visits / self.p_stop_warmup)
            if self.p_stop_warmup > 0
            else 1.0
        )
        return (1.0 - alpha) * self.p_stop_rollout + alpha * p_learned

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to [0, 1] using running min-max.

        Returns 0.5 when reward range is zero, which covers the initial
        rollouts where only one distinct reward has been observed (min == max)
        as well as degenerate cases where all rewards are identical.
        """
        reward_range = self.reward_max - self.reward_min
        if reward_range <= 0:
            return 0.5
        return (reward - self.reward_min) / reward_range

    def _score_rollout_actions(
        self, group_idx: int, legal_actions: list[int]
    ) -> dict[int, float]:
        """Score legal rollout actions using learned statistics.

        For each action, computes:
            score(a) = mean_reward(a) + novelty_weight / sqrt(visits(a) + 1)

        The 1/sqrt(visits) term is a UCB-style exploration bonus that decays as
        an action is visited more, encouraging under-explored actions to be tried.
        Actions with no stats get score = novelty_weight (maximum exploration).

        Args:
            group_idx: Index of the current group
            legal_actions: List of legal action indices (may include STOP)

        Returns:
            Dictionary mapping each action to its score
        """
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

    def _sample_rollout_action(self, group_idx: int, legal_actions: list[int]) -> int:
        """Sample a rollout action using softmax policy blended with uniform.

        Computes p(a) = (1 - epsilon) * softmax(score/tau) + epsilon * uniform.

        Args:
            group_idx: Index of the current group
            legal_actions: List of legal action indices (may include STOP)

        Returns:
            Selected action index
        """
        scores = self._score_rollout_actions(group_idx, legal_actions)
        n = len(legal_actions)

        # Compute softmax probabilities with temperature
        logits = torch.tensor([scores[a] for a in legal_actions], dtype=torch.float64)
        policy_probs = torch.softmax(logits / self.rollout_tau, dim=0)

        # Blend with uniform: p(a) = (1 - eps) * softmax + eps * uniform
        eps = self.rollout_epsilon
        probs = (1.0 - eps) * policy_probs + eps / n

        # Sample using weighted choice
        return self.rng.choices(legal_actions, weights=probs.tolist(), k=1)[0]

    def _update_rollout_stats(
        self, trajectory: list[tuple[int, int, int]], reward: float
    ) -> None:
        """Update rollout policy statistics from a completed trajectory.

        Args:
            trajectory: List of (group_idx, cardinality, action) triples from
                the rollout
            reward: Raw reward obtained from the terminal evaluation
        """
        for group_idx, _cardinality, action in trajectory:
            key = (group_idx, action)
            v, tot = self.rollout_stats.get(key, (0, 0.0))
            self.rollout_stats[key] = (v + 1, tot + reward)

    @staticmethod
    def _extract_tree_actions(path: list[Node]) -> list[tuple[int, int, int]]:
        """Extract (group_idx, cardinality, action) from consecutive node pairs.

        For each parent-child pair in the tree path, determines which action
        was taken (feature selection or STOP) and records the context
        (group index and cardinality at the time of the action).

        Args:
            path: List of nodes from root to leaf in the tree traversal

        Returns:
            List of (group_idx, cardinality, action) triples
        """
        context_actions: list[tuple[int, int, int]] = []
        for i in range(len(path) - 1):
            parent = path[i]
            child = path[i + 1]
            g = parent.group_idx
            cardinality = len(parent.partial_by_group[g])
            if child.stopped_by_group[g] and not parent.stopped_by_group[g]:
                action = STOP
            else:
                child_partial = child.partial_by_group[g]
                parent_partial = parent.partial_by_group[g]
                if len(child_partial) > len(parent_partial):
                    action = child_partial[-1]
                else:
                    continue
            context_actions.append((g, cardinality, action))
        return context_actions

    def _update_context_rave_stats(
        self, context_actions: list[tuple[int, int, int]], reward: float
    ) -> None:
        """Update context-aware RAVE statistics.

        Args:
            context_actions: List of (group_idx, cardinality, action) triples
            reward: Normalized reward to accumulate
        """
        for group_idx, cardinality, action in context_actions:
            key = (group_idx, cardinality, action)
            v, tot = self.context_rave_stats.get(key, (0, 0.0))
            self.context_rave_stats[key] = (v + 1, tot + reward)

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

    def _get_selection(self, node: Node) -> tuple[tuple[int, ...], dict[int, float]]:
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
                # partial[0] is the index into cat_group.values
                cat_selections[cat_group.dim] = cat_group.values[partial[0]]

        return selected_features_tuple, cat_selections

    def _select_and_expand(self) -> tuple[Node, list[Node]]:
        """Select path through tree and expand one new node."""
        node = self.root
        path = [node]

        while not node.is_terminal(self.groups):
            legal = self._legal_actions(node)
            limit = self._child_limit(node)
            unexpanded = [a for a in legal if a not in node.children.keys()]
            can_expand = len(node.children) < limit

            if can_expand and unexpanded:
                # Expand one new child
                action = self.rng.choice(unexpanded)
                child = self._apply_action(node, action)
                node.children[action] = child
                path.append(child)
                return child, path

            # UCT + RAVE selection among existing children
            # Bind node via default argument to avoid B023 closure issue
            def combined_score(action: int, child: Node, _node: Node = node) -> float:
                parent_visits = max(1, _node.n_visits)
                child_visits = max(1, child.n_visits)
                uct_val = (child.w_total / child_visits) + self.c_uct * math.sqrt(
                    math.log(parent_visits) / child_visits
                )

                if self.context_rave:
                    g = _node.group_idx
                    cardinality = len(_node.partial_by_group[g])
                    ctx_key = (g, cardinality, action)
                    v, tot = self.context_rave_stats.get(ctx_key, (0, 0.0))
                    rave_mean = (tot / v) if v > 0 else 0.0
                else:
                    if action == STOP:
                        rave_mean = 0.0
                    else:
                        g = _node.group_idx
                        glob_id = self._global_action_id(g, action)
                        v, tot = self.rave_stats.get(glob_id, (0, 0.0))
                        rave_mean = (tot / v) if v > 0 else 0.0

                beta = self.k_rave / (self.k_rave + max(1, _node.n_visits))
                return (1 - beta) * uct_val + beta * rave_mean

            if node.children:
                best_action, best_child = max(
                    node.children.items(), key=lambda kv: combined_score(kv[0], kv[1])
                )
                node = best_child
                path.append(node)
            else:
                break

        return node, path

    def _rollout(
        self, node: Node
    ) -> tuple[tuple[int, ...], dict[int, float], list[tuple[int, int, int]]]:
        """Random rollout to terminal state, return selection and trajectory.

        Returns:
            Tuple of (selected_features, cat_selections, trajectory) where
            trajectory is a list of (group_idx, cardinality, action) triples
            taken during rollout.
        """
        curr = Node(
            partial_by_group=tuple(node.partial_by_group),
            stopped_by_group=tuple(node.stopped_by_group),
            group_idx=node.group_idx,
        )
        trajectory: list[tuple[int, int, int]] = []

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

            if self.rollout_policy:
                # Learned softmax policy: STOP is scored like any other action
                action = self._sample_rollout_action(g, legal)
            else:
                # Original logic: adaptive p_stop for NChooseK, uniform for features
                is_nchoosek = g < len(self.groups.nchooseks)
                if is_nchoosek and STOP in legal:
                    p_stop = self._compute_adaptive_p_stop(
                        g, len(curr.partial_by_group[g])
                    )
                    if self.rng.random() < p_stop:
                        trajectory.append((g, len(curr.partial_by_group[g]), STOP))
                        curr = self._apply_action(curr, STOP)
                        continue

                # Choose uniformly among non-STOP actions
                choices = [a for a in legal if a != STOP]
                if not choices:
                    trajectory.append((g, len(curr.partial_by_group[g]), STOP))
                    curr = self._apply_action(curr, STOP)
                    continue

                action = self.rng.choice(choices)

            trajectory.append((g, len(curr.partial_by_group[g]), action))
            curr = self._apply_action(curr, action)

        selected_features, cat_selections = self._get_selection(curr)
        return selected_features, cat_selections, trajectory

    def _backpropagate(
        self,
        path: list[Node],
        reward: float,
        selected_features: tuple[int, ...],
        cat_selections: dict[int, float],
    ) -> None:
        """Backpropagate reward through path and update RAVE statistics."""
        for n in path:
            n.n_visits += 1
            n.w_total += reward

        # Update RAVE stats for NChooseK selections
        selected_set = set(selected_features)
        for g, nchoosek in enumerate(self.groups.nchooseks):
            for local_idx, feat_idx in enumerate(nchoosek.features):
                if feat_idx in selected_set:
                    glob_id = self._global_action_id(g, local_idx)
                    v, tot = self.rave_stats.get(glob_id, (0, 0.0))
                    self.rave_stats[glob_id] = (v + 1, tot + reward)

        # Update RAVE stats for categorical selections
        n_nchoosek = len(self.groups.nchooseks)
        for i, cat_group in enumerate(self.groups.categoricals):
            g = n_nchoosek + i
            if cat_group.dim in cat_selections:
                selected_value = cat_selections[cat_group.dim]
                for local_idx, value in enumerate(cat_group.values):
                    if value == selected_value:
                        glob_id = self._global_action_id(g, local_idx)
                        v, tot = self.rave_stats.get(glob_id, (0, 0.0))
                        self.rave_stats[glob_id] = (v + 1, tot + reward)
                        break

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
                trajectory: list[tuple[int, int, int]] = []
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

            # Update reward range (used by normalization and adaptive p_stop)
            if reward < self.reward_min:
                self.reward_min = reward
            if reward > self.reward_max:
                self.reward_max = reward

            if reward > self.best_value:
                self.best_value = reward
                self.best_selection = (selected_features, cat_selections)

            if self.adaptive_p_stop:
                self._update_cardinality_stats(reward, selected_features)

            # Normalize reward for backpropagation if enabled
            bp_reward = (
                self._normalize_reward(reward) if self.normalize_rewards else reward
            )

            if is_novel:
                self._backpropagate(path, bp_reward, selected_features, cat_selections)
                if self.context_rave:
                    tree_actions = self._extract_tree_actions(path)
                    all_context_actions = tree_actions + trajectory
                    self._update_context_rave_stats(all_context_actions, bp_reward)
            else:
                # Virtual loss: increment visits with zero reward so that
                # (a) PW limits still grow with traffic, and
                # (b) mean_value drops for over-visited branches, steering
                #     UCT exploration toward less-exploited parts of the tree.
                for n in path:
                    n.n_visits += 1

            self._update_rollout_stats(trajectory, reward)

        if self.best_selection is None:
            return (), {}, self.best_value
        return self.best_selection[0], self.best_selection[1], self.best_value

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
    # MCTS parameters
    c_uct: float = 0.01,
    k_rave: float = 0.0,
    p_stop_rollout: float = 0.35,
    num_iterations: int = 100,
    pw_k0: float = 2.0,
    pw_alpha: float = 0.6,
    max_rollout_retries: int = 3,
    adaptive_p_stop: bool = True,
    p_stop_warmup: int = 20,
    p_stop_temperature: float = 0.25,
    normalize_rewards: bool = True,
    rollout_policy: bool = True,
    rollout_epsilon: float = 0.3,
    rollout_tau: float = 1.0,
    rollout_novelty_weight: float = 1.0,
    context_rave: bool = False,
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

    Uses MCTS to select which features are active (non-zero) and categorical values,
    then runs BoTorch optimization with inactive features fixed to zero and
    categoricals fixed to their selected values.

    Args:
        acq_function: BoTorch acquisition function to optimize
        bounds: 2 x d tensor of (lower, upper) bounds for each dimension
        nchooseks: List of NChooseK constraints as tuples of (features, min_count, max_count)
            where features is a list of feature indices
        cat_dims: Dictionary mapping categorical dimension indices to allowed values
            (same signature as botorch.optim.optimize_acqf_mixed_alternating)
        c_uct: UCT exploration constant (default 0.01, paired with normalize_rewards)
        k_rave: RAVE blending decay parameter (default 0 = disabled)
        p_stop_rollout: Base probability of early stop during NChooseK rollout
        num_iterations: Number of MCTS iterations
        pw_k0: Progressive widening base constant
        pw_alpha: Progressive widening exponent
        max_rollout_retries: Maximum rollout retries on cache hit
        adaptive_p_stop: Learn per-group stop probability from cardinality stats
        p_stop_warmup: Number of rollouts before adaptive p_stop fully activates
        p_stop_temperature: Sigmoid temperature for adaptive p_stop
        normalize_rewards: Map rewards to [0, 1] via running min-max
        rollout_policy: Use learned softmax rollout policy instead of uniform
        rollout_epsilon: Epsilon for epsilon-greedy blending in rollout policy
        rollout_tau: Temperature for softmax in rollout policy
        rollout_novelty_weight: Novelty bonus coefficient for rollout policy
        context_rave: Use context-aware RAVE instead of global RAVE
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
        c_uct=c_uct,
        k_rave=k_rave,
        p_stop_rollout=p_stop_rollout,
        pw_k0=pw_k0,
        pw_alpha=pw_alpha,
        max_rollout_retries=max_rollout_retries,
        adaptive_p_stop=adaptive_p_stop,
        p_stop_warmup=p_stop_warmup,
        p_stop_temperature=p_stop_temperature,
        normalize_rewards=normalize_rewards,
        rollout_policy=rollout_policy,
        rollout_epsilon=rollout_epsilon,
        rollout_tau=rollout_tau,
        rollout_novelty_weight=rollout_novelty_weight,
        context_rave=context_rave,
        seed=seed,
    )

    mcts.run(n_iterations=num_iterations)

    # Handle case where no valid solution was found
    if best_candidates is None:
        # Return zeros with -inf value
        best_candidates = torch.zeros(q, d, dtype=bounds.dtype, device=bounds.device)
        best_acq_value = float("-inf")

    return best_candidates, best_acq_value
