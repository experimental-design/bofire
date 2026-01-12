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
        c_uct: UCT exploration constant (default 1.0)
        k_rave: RAVE blending decay parameter (default 300.0)
        p_stop_rollout: Probability of early stop during rollout (default 0.35)
        pw_k0: Progressive widening base constant (default 2.0)
        pw_alpha: Progressive widening exponent (default 0.6)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        groups: Groups,
        reward_fn: Callable[[tuple[int, ...], dict[int, float]], float],
        c_uct: float = 1.0,
        k_rave: float = 300.0,
        p_stop_rollout: float = 0.35,
        pw_k0: float = 2.0,
        pw_alpha: float = 0.6,
        seed: Optional[int] = None,
    ):
        self.groups = groups
        self.reward_fn = reward_fn
        self.c_uct = c_uct
        self.k_rave = k_rave
        self.p_stop_rollout = p_stop_rollout
        self.pw_k0 = pw_k0
        self.pw_alpha = pw_alpha
        self.rng = random.Random(seed)

        # Initialize root node
        n_groups = len(groups)
        self.root = Node(
            partial_by_group=tuple(() for _ in range(n_groups)),
            stopped_by_group=tuple(False for _ in range(n_groups)),
            group_idx=0,
        )

        # Best found so far
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

    def _compute_group_offsets(self) -> list[int]:
        """Compute offset for each group to create global action IDs."""
        offsets = []
        acc = 0
        print("Groups:")
        print(self.groups)
        for group in self.groups.groups:
            print(group)
            offsets.append(acc)
            acc += group.n_options
        return offsets

    def _global_action_id(self, group_idx: int, local_idx: int) -> int:
        """Convert (group, local_index) to global action ID for RAVE."""
        return self.global_offsets[group_idx] + local_idx

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
            new_partial = partials[g] + (action,)
            partials[g] = new_partial
            # Check if group is complete
            if group.is_complete(new_partial, stoppeds[g]):
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

                if action == STOP:
                    rave_mean = 0.0
                else:
                    g = _node.group_idx
                    glob_id = self._global_action_id(g, action)
                    v, tot = self.rave_stats.get(glob_id, (0, 0.0))
                    rave_mean = (tot / v) if v > 0 else 0.0

                beta = self.k_rave / (self.k_rave + _node.n_visits)
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

    def _rollout(self, node: Node) -> tuple[tuple[int, ...], dict[int, float]]:
        """Random rollout to terminal state, return selection."""
        curr = Node(
            partial_by_group=tuple(node.partial_by_group),
            stopped_by_group=tuple(node.stopped_by_group),
            group_idx=node.group_idx,
        )

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

            # For NChooseK groups, prefer STOP with some probability
            g = curr.group_idx
            is_nchoosek = g < len(self.groups.nchooseks)
            if (
                is_nchoosek
                and STOP in legal
                and self.rng.random() < self.p_stop_rollout
            ):
                curr = self._apply_action(curr, STOP)
                continue

            # Choose uniformly among non-STOP actions
            choices = [a for a in legal if a != STOP]
            if not choices:
                curr = self._apply_action(curr, STOP)
                continue

            action = self.rng.choice(choices)
            curr = self._apply_action(curr, action)

        return self._get_selection(curr)

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
            else:
                selected_features, cat_selections = self._rollout(leaf)

            reward = self._cached_reward(selected_features, cat_selections)

            if reward > self.best_value:
                self.best_value = reward
                self.best_selection = (selected_features, cat_selections)

            self._backpropagate(path, reward, selected_features, cat_selections)

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
    nchooseks: list[NChooseK] | None = None,
    cat_dims: Mapping[int, Sequence[float]] | None = None,
    # MCTS parameters
    c_uct: float = 1.0,
    k_rave: float = 300.0,
    p_stop_rollout: float = 0.35,
    num_iterations: int = 100,
    pw_k0: float = 2.0,
    pw_alpha: float = 0.6,
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
        nchooseks: Sequence of NChooseK constraints defining feature groups
        cat_dims: Dictionary mapping categorical dimension indices to allowed values
            (same signature as botorch.optim.optimize_acqf_mixed_alternating)
        c_uct: UCT exploration constant
        k_rave: RAVE blending decay parameter
        p_stop_rollout: Probability of early stop during NChooseK rollout
        num_iterations: Number of MCTS iterations
        pw_k0: Progressive widening base constant
        pw_alpha: Progressive widening exponent
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

    # Build constraints
    categorical_list = (
        [Categorical(dim=dim, values=list(values)) for dim, values in cat_dims.items()]
        if cat_dims
        else []
    )
    nchooseks = nchooseks or []
    all_groups = nchooseks + categorical_list
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
        seed=seed,
    )

    mcts.run(n_iterations=num_iterations)

    # Handle case where no valid solution was found
    if best_candidates is None:
        # Return zeros with -inf value
        best_candidates = torch.zeros(q, d, dtype=bounds.dtype, device=bounds.device)
        best_acq_value = float("-inf")

    return best_candidates, best_acq_value
