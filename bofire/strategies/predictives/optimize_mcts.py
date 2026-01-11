"""MCTS-based acquisition function optimization for NChooseK constraints.

Uses Monte Carlo Tree Search to select which features are active (non-zero),
then runs BoTorch acquisition function optimization with inactive features fixed to zero.
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
from botorch.optim import optimize_acqf
from torch import Tensor


STOP = None  # Sentinel for stopping selection in a group


@dataclass(frozen=True)
class NChooseK:
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
    def n_features(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class Constraints:
    """Collection of NChooseK constraints."""

    constraints: list[NChooseK]

    def __len__(self) -> int:
        return len(self.constraints)

    @property
    def all_features(self) -> list[int]:
        all_feats = []
        for c in self.constraints:
            all_feats.extend(c.features)
        return all_feats


@dataclass
class Node:
    """MCTS tree node.

    Args:
        partial_by_group: Partial selection per group (indices into group's feature list)
        stopped_by_group: Whether each group has stopped selecting
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

    children: dict[Union[int, None], "Node"] = field(default_factory=dict)

    def is_terminal(self, constraints: Constraints) -> bool:
        return self.group_idx >= len(constraints)

    def mean_value(self) -> float:
        return self.w_total / self.n_visits if self.n_visits > 0 else 0.0


def legal_actions_group(
    constraint: NChooseK, partial: tuple[int, ...], stopped: bool
) -> list[Union[int, None]]:
    """Compute legal actions within a single NChooseK group.

    Actions are indices into constraint.features (not the actual feature indices).
    Enforces strictly increasing selection order (combinations, not permutations).
    STOP is legal iff len(partial) >= min_count and not already stopped.

    Args:
        constraint: The NChooseK constraint for this group
        partial: Current partial selection (indices into constraint.features)
        stopped: Whether this group has already stopped

    Returns:
        List of legal actions (int indices or STOP sentinel)
    """
    n = constraint.n_features
    m = len(partial)

    if stopped or m >= constraint.max_count:
        return []

    actions: list[Union[int, None]] = []
    last = partial[-1] if partial else -1

    # Remaining picks needed after this action to satisfy min_count
    r_min_needed = max(0, constraint.min_count - (m + 1))
    # After picking index i, n - (i+1) items remain; require n - (i+1) >= r_min_needed
    end_inclusive = n - r_min_needed - 1
    start = last + 1

    if start <= end_inclusive:
        actions.extend(range(start, end_inclusive + 1))

    if m >= constraint.min_count:
        actions.append(STOP)

    return actions


class MCTS:
    """Monte Carlo Tree Search for NChooseK constraint optimization.

    Uses UCT selection, RAVE action value estimation, and progressive widening.
    Selects which features are active via tree search, evaluating terminals
    with a provided reward function.

    Args:
        constraints: Collection of NChooseK constraints
        reward_fn: Function mapping selected feature indices to reward value
        c_uct: UCT exploration constant (default 1.0)
        k_rave: RAVE blending decay parameter (default 300.0)
        p_stop_rollout: Probability of early stop during rollout (default 0.35)
        pw_k0: Progressive widening base constant (default 2.0)
        pw_alpha: Progressive widening exponent (default 0.6)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        constraints: Constraints,
        reward_fn: Callable[[tuple[int, ...]], float],
        c_uct: float = 1.0,
        k_rave: float = 300.0,
        p_stop_rollout: float = 0.35,
        pw_k0: float = 2.0,
        pw_alpha: float = 0.6,
        seed: Optional[int] = None,
    ):
        self.constraints = constraints
        self.reward_fn = reward_fn
        self.c_uct = c_uct
        self.k_rave = k_rave
        self.p_stop_rollout = p_stop_rollout
        self.pw_k0 = pw_k0
        self.pw_alpha = pw_alpha
        self.rng = random.Random(seed)

        # Initialize root node
        n_groups = len(constraints)
        self.root = Node(
            partial_by_group=tuple(() for _ in range(n_groups)),
            stopped_by_group=tuple(False for _ in range(n_groups)),
            group_idx=0,
        )

        # Best found so far
        self.best_features: Optional[tuple[int, ...]] = None
        self.best_value: float = float("-inf")

        # Cache for terminal evaluations
        self.value_cache: dict[tuple[int, ...], float] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # RAVE statistics: global_id -> (visits, total_reward)
        self.global_offsets = self._compute_group_offsets()
        self.rave_stats: dict[int, tuple[int, float]] = {}

    def _compute_group_offsets(self) -> list[int]:
        """Compute offset for each group to create global action IDs."""
        offsets = []
        acc = 0
        for c in self.constraints.constraints:
            offsets.append(acc)
            acc += c.n_features
        return offsets

    def _global_action_id(self, group_idx: int, local_idx: int) -> int:
        """Convert (group, local_index) to global action ID for RAVE."""
        return self.global_offsets[group_idx] + local_idx

    def _cached_reward(self, features: tuple[int, ...]) -> float:
        """Get cached reward or compute and cache it."""
        if features in self.value_cache:
            self.cache_hits += 1
            return self.value_cache[features]
        val = self.reward_fn(features)
        self.value_cache[features] = val
        self.cache_misses += 1
        return val

    def _child_limit(self, node: Node) -> int:
        """Progressive widening: max children based on visit count."""
        return max(1, int(self.pw_k0 * (max(1, node.n_visits) ** self.pw_alpha)))

    def _legal_actions(self, node: Node) -> list[Union[int, None]]:
        """Get legal actions for current group in node."""
        if node.is_terminal(self.constraints):
            return []
        g = node.group_idx
        constraint = self.constraints.constraints[g]
        partial = node.partial_by_group[g]
        stopped = node.stopped_by_group[g]
        return legal_actions_group(constraint, partial, stopped)

    def _apply_action(self, node: Node, action: Union[int, None]) -> Node:
        """Create child node by applying action to current node."""
        g = node.group_idx

        partials = list(node.partial_by_group)
        stoppeds = list(node.stopped_by_group)

        if action is STOP:
            stoppeds[g] = True
            next_g = g + 1
        else:
            new_partial = partials[g] + (action,)
            partials[g] = new_partial
            constraint = self.constraints.constraints[g]
            # Auto-advance if max_count reached
            next_g = g + 1 if len(new_partial) >= constraint.max_count else g

        return Node(
            partial_by_group=tuple(partials),
            stopped_by_group=tuple(stoppeds),
            group_idx=next_g,
        )

    def _get_selected_features(self, node: Node) -> tuple[int, ...]:
        """Convert node's partial selections to actual feature indices."""
        features = []
        for g, constraint in enumerate(self.constraints.constraints):
            for local_idx in node.partial_by_group[g]:
                features.append(constraint.features[local_idx])
        return tuple(sorted(features))

    def _select_and_expand(self) -> tuple[Node, list[Node]]:
        """Select path through tree and expand one new node."""
        node = self.root
        path = [node]

        while not node.is_terminal(self.constraints):
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

            # UCT + RAVE selection among existing children
            # Bind node via default argument to avoid B023 closure issue
            def combined_score(
                action: Union[int, None], child: Node, _node: Node = node
            ) -> float:
                parent_visits = max(1, _node.n_visits)
                child_visits = max(1, child.n_visits)
                uct_val = (child.w_total / child_visits) + self.c_uct * math.sqrt(
                    math.log(parent_visits) / child_visits
                )

                if action is STOP:
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

    def _rollout(self, node: Node) -> tuple[int, ...]:
        """Random rollout to terminal state, return selected feature indices."""
        curr = Node(
            partial_by_group=tuple(node.partial_by_group),
            stopped_by_group=tuple(node.stopped_by_group),
            group_idx=node.group_idx,
        )

        while not curr.is_terminal(self.constraints):
            legal = self._legal_actions(curr)
            if not legal:
                # No legal actions, advance group (happens when max_count reached)
                curr = Node(
                    partial_by_group=curr.partial_by_group,
                    stopped_by_group=curr.stopped_by_group,
                    group_idx=curr.group_idx + 1,
                )
                continue

            # Prefer STOP with some probability
            if STOP in legal and self.rng.random() < self.p_stop_rollout:
                curr = self._apply_action(curr, STOP)
                continue

            # Choose uniformly among non-STOP actions
            choices = [a for a in legal if a is not STOP]
            if not choices:
                curr = self._apply_action(curr, STOP)
                continue

            action = self.rng.choice(choices)
            curr = self._apply_action(curr, action)

        return self._get_selected_features(curr)

    def _backpropagate(
        self, path: list[Node], reward: float, selected_features: tuple[int, ...]
    ) -> None:
        """Backpropagate reward through path and update RAVE statistics."""
        for n in path:
            n.n_visits += 1
            n.w_total += reward

        # Update RAVE stats for all selected features
        selected_set = set(selected_features)
        for g, constraint in enumerate(self.constraints.constraints):
            for local_idx, feat_idx in enumerate(constraint.features):
                if feat_idx in selected_set:
                    glob_id = self._global_action_id(g, local_idx)
                    v, tot = self.rave_stats.get(glob_id, (0, 0.0))
                    self.rave_stats[glob_id] = (v + 1, tot + reward)

    def run(self, n_iterations: int) -> tuple[tuple[int, ...], float]:
        """Run MCTS for specified number of iterations.

        Args:
            n_iterations: Number of MCTS iterations to run

        Returns:
            Tuple of (best_features, best_value) found during search
        """
        for _ in range(n_iterations):
            leaf, path = self._select_and_expand()

            if leaf.is_terminal(self.constraints):
                selected_features = self._get_selected_features(leaf)
            else:
                selected_features = self._rollout(leaf)

            reward = self._cached_reward(selected_features)

            if reward > self.best_value:
                self.best_value = reward
                self.best_features = selected_features

            self._backpropagate(path, reward, selected_features)

        return (self.best_features or ()), self.best_value

    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "size": len(self.value_cache),
        }


def optimize_acqf_mcts(
    acq_function,
    bounds: Tensor,
    nchooseks: Sequence[NChooseK],
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
    """Optimize acquisition function with NChooseK constraints using MCTS.

    Uses MCTS to select which features are active (non-zero), then runs BoTorch
    optimization with inactive features fixed to zero.

    Args:
        acq_function: BoTorch acquisition function to optimize
        bounds: 2 x d tensor of (lower, upper) bounds for each dimension
        nchooseks: Sequence of NChooseK constraints defining feature groups
        cat_dims: Categorical dimensions mapping (ignored for now)
        c_uct: UCT exploration constant
        k_rave: RAVE blending decay parameter
        p_stop_rollout: Probability of early stop during rollout
        num_iterations: Number of MCTS iterations
        pw_k0: Progressive widening base constant
        pw_alpha: Progressive widening exponent
        q: Batch size for acquisition function optimization
        raw_samples: Number of raw samples for initialization
        num_restarts: Number of optimization restarts
        fixed_features: Additional fixed features (combined with inactive ones)
        inequality_constraints: Inequality constraints for BoTorch optimization
        equality_constraints: Equality constraints for BoTorch optimization
        seed: Random seed for reproducibility

    Returns:
        Tuple of (best_candidates, best_acq_value) where best_candidates is a
        q x d tensor of optimal points and best_acq_value is the acquisition value
    """
    d = bounds.shape[1]
    constraints = Constraints(constraints=list(nchooseks))

    # All feature indices covered by NChooseK constraints
    nchoosek_features = set(constraints.all_features)

    # Storage for best result across all MCTS evaluations
    best_candidates: Optional[Tensor] = None
    best_acq_value: float = float("-inf")

    def reward_fn(selected_features: tuple[int, ...]) -> float:
        nonlocal best_candidates, best_acq_value

        selected_set = set(selected_features)

        # Build fixed_features dict: inactive NChooseK features fixed to 0
        inactive_features = nchoosek_features - selected_set

        combined_fixed: dict[int, float] = {}
        # First add user-provided fixed features
        if fixed_features is not None:
            combined_fixed.update(fixed_features)
        # Then fix inactive features to 0
        for idx in inactive_features:
            combined_fixed[idx] = 0.0

        # Run BoTorch optimization
        try:
            candidates, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                fixed_features=combined_fixed if combined_fixed else None,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
            )

            value = acq_value.item()

            # Track best
            if value > best_acq_value:
                best_acq_value = value
                best_candidates = candidates

            return value

        except Exception:
            # If optimization fails, return very negative value
            return float("-inf")

    # Run MCTS
    mcts = MCTS(
        constraints=constraints,
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
