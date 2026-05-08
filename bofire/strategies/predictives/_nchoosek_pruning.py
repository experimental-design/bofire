"""Greedy NChooseK pruning, based on the BONSAI algorithm
(https://arxiv.org/abs/2602.07144).

This module is a self-contained, side-effect-free implementation of
the greedy pruning post-processing step that turns the output of an
unconstrained acquisition function maximisation into a candidate that
satisfies all NChooseK constraints and per-feature semi-continuity
constraints (``allow_zero=True`` with ``lb > 0``).

The full conceptual write-up lives in ``docs/pruning.md``. The module
operates on tensors throughout — the caller is responsible for
``domain.inputs.transform`` / ``inverse_transform``, and for building
the ``features2idx``, ``bounds`` and linear-constraint tensors that
the algorithm consumes.

The module assumes that the columns of ``X``/``bounds`` are aligned
with the indices reported by ``features2idx`` and used inside
``inequality_constraints`` / ``equality_constraints``. For
pure-continuous domains (Phase 1's scope) this holds by construction;
mixing NChooseK with one-hot-encoded categorical features is out of
scope.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.optim.parameter_constraints import project_to_feasible_space_via_slsqp
from torch import Tensor

from bofire.data_models.constraints.api import (
    InterpointConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearConstraint,
    ProductConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput


class PruningInfeasibleError(RuntimeError):
    """Raised when greedy pruning cannot satisfy every NChooseK and
    semi-continuity constraint — typically because the per-constraint
    ``min_count`` guard empties the action set before all
    ``max_count`` constraints are met.
    """


class ActionKind(Enum):
    """The three primitive action kinds of the greedy pruning loop.

    Values are integers ordered by the existing tie-break preference
    (``ZERO < ACTIVE < ACTIVATE``) so ``(af_red, kind.value, j)`` is a
    drop-in replacement for the legacy ``(af_red, 0/1, j)`` ordering.
    """

    ZERO = 0
    ACTIVE = 1
    ACTIVATE = 2


@dataclass
class Action:
    """A single greedy-pruning move proposed for selection.

    ``valid`` is ``False`` when the per-action variant builder failed
    (typically a QP-projection failure on mutually infeasible bounds +
    linear constraints); the caller scores such variants at ``-inf``
    so they are never selected.
    """

    j: int
    kind: ActionKind
    variant: Tensor
    valid: bool


@dataclass(frozen=True)
class PruningContext:
    """All static-per-call inputs to the pruning loop.

    Bundling these reduces the parameter-count of the inner helpers
    (variant builders, action collectors) from ~10 to ~3 and gives
    a single source of truth for the per-call configuration.
    """

    bounds: Tensor
    inequality_constraints: List[Tuple[Tensor, Tensor, float]]
    equality_constraints: List[Tuple[Tensor, Tensor, float]]
    acqf: AcquisitionFunction
    semicontinuous_specs: Dict[int, Tuple[float, float]]
    fixed_features: Optional[Dict[int, float]]
    nchoosek_constraints: Sequence[NChooseKConstraint]
    features2idx: Dict[str, Tuple[int, ...]]
    tol: float
    per_step_local_reopt: bool


@dataclass
class PruningState:
    """Per-candidate mutable state inside the greedy loop.

    ``zero_set``, ``frac_set``, and ``active_set`` partition the tensor
    column indices by their current (zero / fractional / active)
    classification. ``commit`` applies an action's effect on the
    partition; the candidate tensor ``x`` itself is updated by the
    caller from ``Action.variant``.
    """

    x: Tensor
    zero_set: Set[int] = field(default_factory=set)
    frac_set: Set[int] = field(default_factory=set)
    active_set: Set[int] = field(default_factory=set)

    def commit(self, action: Action) -> None:
        """Apply ``action``'s effect on the (zero, frac, active) partition.

        Mirrors the legacy in-place update at ``prune_nchoosek``'s
        ``kind_pick`` if/elif/else block.
        """
        j = action.j
        if action.kind is ActionKind.ZERO:
            self.frac_set.discard(j)
            self.active_set.discard(j)
            self.zero_set.add(j)
        elif action.kind is ActionKind.ACTIVE:
            self.frac_set.discard(j)
            self.active_set.add(j)
        else:  # ActionKind.ACTIVATE
            self.zero_set.discard(j)
            self.active_set.add(j)


# ---------------------------------------------------------------------------
# Domain-level applicability gates
# ---------------------------------------------------------------------------


def has_semicontinuous_features(domain: Domain) -> bool:
    """True iff any continuous input has ``allow_zero=True`` and a positive
    lower bound — i.e., its feasible region per coordinate is the
    disconnected union ``{0} ∪ [lb, ub]``.
    """
    for feat in domain.inputs.get(ContinuousInput):
        assert isinstance(feat, ContinuousInput)
        if feat.allow_zero and feat.bounds[0] > 0:
            return True
    return False


def has_nchoosek_linear_overlap(domain: Domain) -> bool:
    """True iff any NChooseK feature also appears in a linear (equality or
    inequality) constraint. Used to determine whether QP projection is
    needed during pruning.
    """
    nchoosek_features: Set[str] = set()
    for c in domain.constraints.get(NChooseKConstraint):
        assert isinstance(c, NChooseKConstraint)
        nchoosek_features.update(c.features)

    linear_features: Set[str] = set()
    for c in domain.constraints.get(
        includes=[LinearEqualityConstraint, LinearInequalityConstraint]
    ):
        linear_features.update(c.features)

    return bool(nchoosek_features.intersection(linear_features))


def _features_in_blocking_constraints(domain: Domain) -> Set[str]:
    blocking: Set[str] = set()
    for c in domain.constraints.get(
        includes=[ProductConstraint, NonlinearConstraint, InterpointConstraint]
    ):
        blocking.update(c.features)
    return blocking


def is_nchoosek_pruning_applicable(domain: Domain) -> bool:
    """True iff greedy pruning can be safely applied for the domain's
    NChooseK constraints (BONSAI algorithm).

    Pruning is applicable when at least one NChooseK constraint exists and
    no NChooseK feature appears in any nonlinear (Product, Nonlinear) or
    interpoint constraint. Overlap with linear equality/inequality
    constraints is allowed and handled via QP projection inside the
    pruning loop.
    """
    nchoosek_constraints = domain.constraints.get(NChooseKConstraint)
    if len(nchoosek_constraints) == 0:
        return False

    blocking = _features_in_blocking_constraints(domain)
    for c in nchoosek_constraints:
        assert isinstance(c, NChooseKConstraint)
        if blocking.intersection(c.features):
            return False
    return True


def semicontinuous_specs_from_domain(
    domain: Domain,
    features2idx: Dict[str, Tuple[int, ...]],
) -> Dict[int, Tuple[float, float]]:
    """Build the ``semicontinuous_specs`` mapping for ``prune_nchoosek``.

    Returns a dict ``{tensor_column_index: (lb, ub)}`` for every continuous
    input with ``allow_zero=True`` and a positive lower bound.
    """
    specs: Dict[int, Tuple[float, float]] = {}
    for feat in domain.inputs.get(ContinuousInput):
        assert isinstance(feat, ContinuousInput)
        if feat.allow_zero and feat.bounds[0] > 0:
            for col in features2idx[feat.key]:
                specs[col] = (float(feat.bounds[0]), float(feat.bounds[1]))
    return specs


def is_pruning_applicable(domain: Domain) -> bool:
    """Unified gate: pruning runs if either NChooseK pruning is
    applicable, or the domain has standalone semi-continuous features —
    and no semi-continuous feature appears in a blocking nonlinear /
    interpoint constraint.
    """
    if is_nchoosek_pruning_applicable(domain):
        return True
    if not has_semicontinuous_features(domain):
        return False

    blocking = _features_in_blocking_constraints(domain)
    for feat in domain.inputs.get(ContinuousInput):
        assert isinstance(feat, ContinuousInput)
        if feat.allow_zero and feat.bounds[0] > 0 and feat.key in blocking:
            return False
    return True


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _classify_features_for_row(
    x: Tensor,
    semicontinuous_specs: Dict[int, Tuple[float, float]],
    tol: float,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Partition tensor column indices into (zero, fractional, active).

    A column is *fractional* when the feature is semi-continuous
    (``j ∈ semicontinuous_specs``) and the current value lies in the
    open interval ``(0, lb_j)``. *Zero* and *active* are defined for
    every column.
    """
    d = x.shape[0]
    zero: Set[int] = set()
    fractional: Set[int] = set()
    active: Set[int] = set()
    for j in range(d):
        v = float(x[j].abs().item())
        if v <= tol:
            zero.add(j)
            continue
        if j in semicontinuous_specs:
            lb_j, _ = semicontinuous_specs[j]
            if v < lb_j - tol:
                fractional.add(j)
                continue
        active.add(j)
    return zero, fractional, active


def _is_nchoosek_fulfilled(
    x: Tensor,
    nchoosek_constraints: Sequence[NChooseKConstraint],
    features2idx: Dict[str, Tuple[int, ...]],
    tol: float = 1e-6,
) -> bool:
    """Tensor-native NChooseK fulfilment check for a single candidate.

    Ported from ``BotorchStrategy._nchoosek_fulfilled_tensor``. Counts
    non-zero columns per constraint (using ``|x[idx]| > tol``) and
    honours ``none_also_valid`` when the count is zero.
    """
    for c in nchoosek_constraints:
        indices: List[int] = []
        for feat_key in c.features:
            indices.extend(features2idx[feat_key])
        count = int((x[indices].abs() > tol).sum().item())
        if count > c.max_count:
            return False
        if count < c.min_count and not (c.none_also_valid and count == 0):
            return False
    return True


def _active_counts(
    x: Tensor,
    nchoosek_constraints: Sequence[NChooseKConstraint],
    features2idx: Dict[str, Tuple[int, ...]],
    tol: float,
) -> Dict[int, int]:
    """Per-constraint active counts, keyed by constraint position."""
    counts: Dict[int, int] = {}
    for c_idx, c in enumerate(nchoosek_constraints):
        indices: List[int] = []
        for feat_key in c.features:
            indices.extend(features2idx[feat_key])
        counts[c_idx] = int((x[indices].abs() > tol).sum().item())
    return counts


def _violated_constraints(
    active_counts: Dict[int, int],
    nchoosek_constraints: Sequence[NChooseKConstraint],
) -> Set[int]:
    """Constraints whose count exceeds ``max_count`` (positions only)."""
    violated: Set[int] = set()
    for c_idx, c in enumerate(nchoosek_constraints):
        if active_counts[c_idx] > c.max_count:
            violated.add(c_idx)
    return violated


def _zero_action_blocked_by_min_count(
    j_idx: int,
    active_counts: Dict[int, int],
    nchoosek_constraints: Sequence[NChooseKConstraint],
    features2idx: Dict[str, Tuple[int, ...]],
) -> bool:
    """True iff zeroing ``j_idx`` would push some ``a_c`` below
    ``min_count_c`` for any constraint ``c`` containing it.

    ``none_also_valid`` exempts the case where the post-commit count
    is exactly zero.
    """
    for c_idx, c in enumerate(nchoosek_constraints):
        constraint_indices: Set[int] = set()
        for feat_key in c.features:
            constraint_indices.update(features2idx[feat_key])
        if j_idx not in constraint_indices:
            continue
        post = active_counts[c_idx] - 1
        if post < c.min_count and not (c.none_also_valid and post == 0):
            return True
    return False


def _activate_action_blocked_by_max_count(
    j_idx: int,
    active_counts: Dict[int, int],
    nchoosek_constraints: Sequence[NChooseKConstraint],
    features2idx: Dict[str, Tuple[int, ...]],
) -> bool:
    """True iff activating ``j_idx`` would push some ``a_c`` above
    ``max_count_c`` for any constraint ``c`` containing it.

    Symmetric of ``_zero_action_blocked_by_min_count`` — prevents
    activating into a max-count violation. No ``none_also_valid``
    carve-out (max_count is a hard ceiling).
    """
    for c_idx, c in enumerate(nchoosek_constraints):
        constraint_indices: Set[int] = set()
        for feat_key in c.features:
            constraint_indices.update(features2idx[feat_key])
        if j_idx not in constraint_indices:
            continue
        if active_counts[c_idx] + 1 > c.max_count:
            return True
    return False


def _min_count_violated_constraints(
    active_counts: Dict[int, int],
    nchoosek_constraints: Sequence[NChooseKConstraint],
) -> Set[int]:
    """Constraints with ``a_c < min_count_c`` (positions only).

    Honours ``none_also_valid``: if a constraint allows
    ``count == 0`` and the current count is 0, it is *not* reported
    as violated.
    """
    violated: Set[int] = set()
    for c_idx, c in enumerate(nchoosek_constraints):
        a_c = active_counts[c_idx]
        if a_c < c.min_count and not (c.none_also_valid and a_c == 0):
            violated.add(c_idx)
    return violated


def _features_eligible_for_zero(
    fractional: Set[int],
    active: Set[int],
    violated: Set[int],
    nchoosek_constraints: Sequence[NChooseKConstraint],
    features2idx: Dict[str, Tuple[int, ...]],
) -> Set[int]:
    """Indices admissible as targets of a ``zero`` action.

    Every fractional feature is admissible (regardless of whether it
    sits in a violated NChooseK), since the algorithm must resolve
    semi-continuity. Active features are admissible only if they
    participate in at least one currently violated constraint —
    zeroing a feature outside every violated constraint costs AF
    without contributing to feasibility.
    """
    eligible: Set[int] = set(fractional)
    if not violated:
        return eligible

    violated_indices: Set[int] = set()
    for c_idx in violated:
        c = nchoosek_constraints[c_idx]
        for feat_key in c.features:
            violated_indices.update(features2idx[feat_key])
    eligible.update(active.intersection(violated_indices))
    return eligible


def _features_eligible_for_activate(
    zero_set: Set[int],
    min_count_violated: Set[int],
    nchoosek_constraints: Sequence[NChooseKConstraint],
    features2idx: Dict[str, Tuple[int, ...]],
) -> Set[int]:
    """Indices admissible as targets of an ``activate`` action.

    Only currently-zero features (``j ∈ zero_set``) that participate
    in at least one min-count-violated NChooseK constraint are
    candidates. Mirror of ``_features_eligible_for_zero`` on the
    zero set, intersected with the min-count-violation feature set.
    """
    if not min_count_violated:
        return set()

    violated_indices: Set[int] = set()
    for c_idx in min_count_violated:
        c = nchoosek_constraints[c_idx]
        for feat_key in c.features:
            violated_indices.update(features2idx[feat_key])
    return zero_set.intersection(violated_indices)


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------
#
# Both variant builders accept the loop's current `active_set` and the
# `semicontinuous_specs` mapping. Before the QP projection or
# `optimize_acqf` call, they tighten bounds on every column in
# `(active_set ∩ semicontinuous_specs.keys()) − {j_idx}` to its
# `[lb_i, ub_i]` band:
#
#     for i in (active_set & semicontinuous_specs.keys()) - {j_idx}:
#         tightened[0, i] = lb_i
#         tightened[1, i] = ub_i
#
# This prevents previously-committed active semi-continuous features
# from drifting back into their gap `(0, lb_i)` when the optimiser
# redistributes mass to satisfy shared linear constraints (mixture
# `Σ x = 1`, etc.). Without this, the per-step reopt for action
# `active(j_{k+1})` could push `x_{j_k}` (committed active in iter k)
# below its own `lb`, leaving the state tracker and tensor
# inconsistent until `_final_local_reopt` cleaned it up.
#
# The `− {j_idx}` exclusion preserves the ability to deactivate a
# previously-active feature. If `j_idx ∈ active_set` and is itself
# semi-continuous, we want it pinned to 0 via `fixed_features`, not
# held in `[lb_j, ub_j]` — those two would conflict. The exclusion
# means "the feature being acted on is exempt from the tightening,
# because the action itself defines its target value/bounds."
#
# Consequences:
#   - per-step reopt is internally consistent across iterations,
#   - `_final_local_reopt` is now genuinely a polish (not a
#     feasibility rescue), so `final_local_reopt`'s role is
#     well-defined regardless of whether semi-continuous features
#     are in play.
# ---------------------------------------------------------------------------
#
# Activate-zero: a third action category for currently-zero features,
# fired when some NChooseK constraint has `a_c < min_count_c` and
# would otherwise leave the loop stuck.
#
# The action set per iteration is:
#   - zero(j)     for j ∈ eligible_for_zero (existing logic),
#   - active(j)   for j ∈ fractional ∩ semicontinuous_specs (existing),
#   - activate(j) for j ∈ zero_set ∩ features(c) for some
#                  min-count-violated c (new).
#
# `_build_activate_variant` is a thin wrapper around
# `_build_active_variant`. It picks the activation target band:
# semi-continuous features get their natural [lb_j, ub_j]; everything
# else gets [max(tol, bounds[0, j_idx]), bounds[1, j_idx]] — any
# positive value within the feature's original bounds, with `tol`
# ensuring the variant is classified as "active" by the |x_j| > tol
# fulfilment rule. The caller passes
# `pinned_zero_indices = zero_set − {j_idx}` so the rest of the zero
# set stays pinned.
#
# `none_also_valid=True`: the loop always activates when
# `0 < count < min_count`. It does not attempt to walk down to
# count=0 by zeroing remaining actives — the per-step
# min-count guard (`_zero_action_blocked_by_min_count`) blocks any
# zero action whose post-state would be in the (0, min_count) gap,
# and the greedy doesn't have multi-step lookahead to plan a
# zero-down trajectory. count=0 is reached only if the AF maximiser
# places the candidate there to begin with; in that case
# none_also_valid=True and the loop accepts it as feasible
# immediately. If a user wants the loop to actively prefer count=0,
# that's an AF-level concern (sparsity prior), not a pruning concern.
#
# Iteration cap: activate makes the action set non-monotone (zero →
# active → zero is possible across iterations if AF preferences
# shift). The loop caps per-candidate iterations at 2 × n_features
# and raises `PruningInfeasibleError` with a clear message if
# exceeded. In practice the AF-driven greedy converges in ≤ d
# iterations because the AF reduction shrinks each step; the cap is
# a safety net against pathological non-determinism.
# ---------------------------------------------------------------------------
#
# Future improvement — swap actions:
#
# Add a fourth action kind `swap(j, k) = zero(j) + activate(k)`
# committed atomically in one iteration. `j ∈ active_set`,
# `k ∈ zero_set`, both in (at least one) shared NChooseK group. Net
# count change is zero per shared constraint, so this is the natural
# "rebalance the support set without changing cardinality" move.
#
# Motivation: the current greedy can only change cardinality. Once a
# feature is committed active it is sticky — there is no path back to
# zero unless a max_count violation reopens it. On
# mixture + NChooseK with tight `min_count = max_count`, the support
# set is locked by the dense AF maximiser even when the GP posterior
# would prefer a different support of the same size. Concrete case:
# dense AF `x = (0.40, 0.35, 0.10, 0.10, 0.05)`, NChooseK over all
# five features with `min=max=3`. Greedy zeros `x5` then `x4` and
# returns support `{1, 2, 3}`. The AF may actually peak at
# `{1, 2, 4}` after redistribution, but reaching it would require
# `zero(3) + activate(4)` together, which the per-step min_count
# guard blocks.
#
# Sketch:
#   - Eligibility per (j, k):
#       * for every NChooseK c with {j,k} ⊆ c.features: count
#         unchanged, always admissible.
#       * j ∈ c, k ∉ c: count_c -= 1; gate with the existing
#         min_count guard.
#       * k ∈ c, j ∉ c: count_c += 1; gate with the max_count guard.
#       * neither in c: no effect.
#   - Variant: tighten bounds with `x_j = 0` pinned and
#     `x_k ∈ [lb_k, ub_k]` (or `[2 * tol, ub_k]` for
#     non-semi-continuous), project onto the linear set, optionally
#     local-reopt — same machinery as the existing variant builders.
#   - Selection rule unchanged: swap competes with zero / active /
#     activate on smallest AF reduction.
#
# Cost: action set grows from O(d) to O(d²) per iteration. For
# d ≤ 30 still cheap (one QP each), worth pre-filtering by AF
# gradient if profiling shows it dominates. Termination is still
# bounded by the 2·d cap.
#
# Where it would pay off: mixture + NChooseK with non-trivial
# `min_count` (formulation problems), and semi-continuous mixtures
# where activation order currently locks in suboptimal supports.
# Spurious-features case probably sees only marginal gains.
# ---------------------------------------------------------------------------
#
# Future improvement — beam search:
#
# The current loop is structurally beam search with width k=1: at each
# iteration it expands the current state into all admissible action
# variants, ranks by AF reduction, and commits the argmin. Widening
# this to k>1 is a natural expansion of BONSAI — the action set, the
# guards, the variant builders, the fulfilment check, and the 2·d
# iteration cap all transfer unchanged. Only the selection rule
# changes from `argmin` to `top-k`.
#
# Refactor sketch:
#   1. Factor the per-iteration body of `prune_nchoosek` into
#      `expand(state) → list[(state', af_reduction)]` that returns
#      every admissible successor (not just the argmin). No
#      behaviour change at k=1.
#   2. Replace the inner `while True / argmin` with a beam loop:
#      hold `beam: list[state]`, expand each, concatenate the
#      successors, retain the top-k by *cumulative* AF reduction.
#   3. Track cumulative AF reduction per beam slot rather than
#      committing in place to `X[i]`.
#   4. On termination, return the best feasible state across the
#      beam.
#   5. Expose `k` as an optional hyperparameter alongside
#      `per_step_local_reopt` / `final_local_reopt`, defaulting to
#      `1` so existing behaviour and tests are unchanged.
#
# Cost: per iteration scales as `k · |actions|`; total work
# `O(k · d²)` (or `O(k · d³)` with swap). Polynomial in d for fixed
# k. Beam-k vs. greedy is a constant-factor `k` slowdown.
#
# What it catches that greedy misses: trajectories where the local
# AF-best move is suboptimal but a continuation from the second-best
# is better. Concretely the "redistribute mass to a different active
# feature" case where the greedy locks in a sticky support set.
#
# What it does *not* catch: trajectories that pass through
# individually-infeasible intermediate states (e.g., `zero(j)` that
# violates min_count even though a downstream `activate(k)` would
# restore feasibility). The per-step guards prune those branches
# before the beam ever sees them — that's exactly the gap that the
# atomic swap action closes. Beam search and swap are therefore
# complementary, not equivalent: swap extends the action *set*,
# beam extends the search *strategy* over that set.
#
# A further structural step beyond plain beam is "lookahead beam":
# admit individually-infeasible states into the beam in the hope
# that a k_lookahead-step continuation restores feasibility. This
# generalises the swap action to arbitrary k, but gives up the
# per-step feasibility invariant and probably needs B&B-style
# bounding to be tractable. Not recommended without that.
# ---------------------------------------------------------------------------
#
# Future improvement — branch-and-bound (B&B):
#
# B&B is the next step beyond beam search. The connection to what we
# already have is direct:
#
#   - Greedy   = beam search with width 1: argmin AF reduction at
#                each step, no backtracking.
#   - Beam-k   = top-k continuations retained per step, ranked by
#                cumulative AF reduction. No proof of optimality, but
#                catches sticky-support trajectories the greedy
#                misses.
#   - B&B      = a priority-queue-driven tree search that retains
#                *all* unpruned partial trajectories. A bounding
#                function `L(state)` provides a lower bound on the
#                AF reduction of any feasible descendant; subtrees
#                whose `accumulated + L(state) >= incumbent` are
#                discarded. With a tight bound the search is exact;
#                with a loose bound it gracefully degrades to a more
#                exhaustive beam-like search.
#
# Why it is a natural next step rather than a rewrite:
#
#   - Branching is the same `expand(state)` function that beam
#     search needs. The action set, guards, and variant builders
#     transfer unchanged.
#   - The current greedy plays four load-bearing roles inside B&B:
#       (a) **Initial incumbent.** Seed `incumbent =
#           greedy(state).af_reduction` so the search has something
#           to prune against from iteration 1. Without this, B&B is
#           effectively unbounded best-first search until the first
#           feasible leaf is reached.
#       (b) **Primal heuristic at every interior node.** Run the
#           greedy from the node to a feasible leaf, update the
#           incumbent if the result is better. Same role the
#           rollout policy plays in MCTS.
#       (c) **Action machinery.** The eligibility, guards, and
#           variant builders are the primitives B&B branches over.
#       (d) **Runtime fallback.** When a node-budget is exceeded,
#           return the incumbent (typically the greedy result) plus
#           the gap `incumbent - best_open_bound` as a quality
#           stamp.
#   - Beam search reuses (c). B&B reuses (a)-(d). Neither obsoletes
#     the greedy.
#
# Reasonable expansion path from the current state of the codebase:
#
#   step 1: factor the per-iteration body of `prune_nchoosek` into
#           an `expand(state) → list[(state', af_reduction)]`
#           function returning every admissible successor (not just
#           the argmin). No behaviour change at width 1; this is
#           the structural prerequisite for everything that follows.
#   step 2: replace the inner `argmin` selection with `top-k`,
#           threaded through a beam list. Add a hyperparameter `k`
#           defaulting to 1 so existing tests are unchanged.
#   step 3: replace the beam list with a priority queue keyed on
#           `accumulated + L(state)`, where `L(state)` is a
#           bounding function. Seed `incumbent` from the greedy at
#           the root. Run the greedy as a primal heuristic at every
#           expanded node and update `incumbent` from its result.
#           Prune subtrees whose bound exceeds `incumbent`. Add a
#           `node_budget` hyperparameter; on exhaustion, return the
#           incumbent with the current gap.
#
# Each step is self-contained: `step 1` is a pure refactor; after
# `step 2` users can opt into beam search; after `step 3` users can
# opt into anytime B&B. The greedy remains the default code path
# throughout.
#
# The bounding function `L(state)` deserves its own design pass and
# is intentionally not specified here. The cheap-and-always-valid
# fallback is `L = 0` (B&B prunes only on accumulated AF reduction);
# tighter bounds use the constraint structure of the violated
# NChooseK group.
# ---------------------------------------------------------------------------


_OPTIMIZE_ACQF_DEFAULTS: Dict[str, Any] = {
    "q": 1,
    "num_restarts": 1,
    "raw_samples": None,
}


def _local_optacqf(
    initial: Tensor,
    bounds: Tensor,
    fixed_features: Optional[Dict[int, float]],
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
    equality_constraints: List[Tuple[Tensor, Tensor, float]],
    acqf: AcquisitionFunction,
) -> Optional[Tensor]:
    """Single-restart ``optimize_acqf`` warm-started from ``initial``.

    Returns the refined ``(d,)`` tensor on success or ``None`` on
    optimizer failure (caller falls back to ``initial``).
    """
    try:
        local_candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            batch_initial_conditions=initial.unsqueeze(0).unsqueeze(0),
            fixed_features=fixed_features,
            inequality_constraints=inequality_constraints
            if inequality_constraints
            else None,
            equality_constraints=equality_constraints if equality_constraints else None,
            **_OPTIMIZE_ACQF_DEFAULTS,
        )
    except Exception:
        return None
    return local_candidate.squeeze(0).squeeze(0)


def _build_variant(
    x_i: Tensor,
    j_idx: int,
    kind: ActionKind,
    *,
    bounds: Tensor,
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
    equality_constraints: List[Tuple[Tensor, Tensor, float]],
    acqf: AcquisitionFunction,
    per_step_local_reopt: bool,
    pinned_zero_indices: Optional[Set[int]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    active_set: Optional[Set[int]] = None,
    semicontinuous_specs: Optional[Dict[int, Tuple[float, float]]] = None,
    tol: float = 1e-6,
) -> Tuple[Tensor, bool]:
    """Construct the candidate variant for one greedy-pruning action.

    Single dispatch over the three :class:`ActionKind`s. The shared
    skeleton (committed-active bound tightening, fixed-pin merging,
    no-linear / SLSQP branch, optional local re-optimisation) is
    consolidated; the per-kind difference is only:

    - ``ZERO``: pin ``x_{j_idx} = 0`` via ``fixed_features``; no bound
      change on ``j_idx``.
    - ``ACTIVE``: tighten ``bounds[:, j_idx] = [lb_j, ub_j]`` from
      ``semicontinuous_specs[j_idx]``; clip starting ``x_{j_idx}`` up
      to ``lb_j``. Caller guarantees ``j_idx`` is fractional and
      semi-continuous.
    - ``ACTIVATE``: tighten ``bounds[:, j_idx]`` to ``[lb_j, ub_j]``
      (semi-continuous case) or ``[max(2*tol, bounds[0, j_idx]),
      bounds[1, j_idx]]`` (non-semi-continuous), so the projected
      variant is classified as active by the ``|x_j| > tol`` rule.

    ``pinned_zero_indices`` are tensor columns committed to zero by
    earlier greedy iterations; they are passed as additional
    ``fixed_features`` so the QP cannot resurrect them. For
    ``ACTIVATE`` actions the caller passes ``zero_set − {j_idx}`` so
    every *other* committed-zero feature stays pinned without
    conflicting with the activation target.

    ``active_set`` ∩ ``semicontinuous_specs`` (excluding ``j_idx``)
    has its bounds tightened to the per-feature semi-continuous band
    so previously-committed active semi-continuous features cannot
    drift into the gap ``(0, lb_i)`` while satisfying the linear
    constraints. The ``− {j_idx}`` exclusion preserves the ability
    to deactivate a previously-active semi-continuous feature.

    ``fixed_features`` is the caller-supplied mapping of features
    that must remain at their fixed values throughout pruning. They
    are merged into the projection's and local re-optimiser's
    ``fixed_features``.

    Returns ``(variant, valid)``:

    - ``valid=True`` if the variant is a feasible point (linear-
      constraint compatible);
    - ``valid=False`` if the QP projection failed (mutually
      infeasible bounds + linear constraints). The variant is still
      returned (with the per-action effect applied as best as
      possible) so the caller has a well-formed tensor; the caller
      must replace its AF value with ``-inf`` so it is never
      selected.
    """
    # Per-kind diff: target bounds on j_idx and whether j_idx is
    # pinned to zero via fixed_features.
    target_lb_j: Optional[float]
    target_ub_j: Optional[float]
    if kind is ActionKind.ZERO:
        target_lb_j = None
        target_ub_j = None
        pin_j_to_zero = True
    elif kind is ActionKind.ACTIVE:
        # Caller guarantees j_idx ∈ frac_set ∩ semicontinuous_specs.
        assert (
            semicontinuous_specs is not None and j_idx in semicontinuous_specs
        ), "ACTIVE action requires j_idx ∈ semicontinuous_specs"
        target_lb_j, target_ub_j = semicontinuous_specs[j_idx]
        pin_j_to_zero = False
    else:  # ActionKind.ACTIVATE
        if semicontinuous_specs and j_idx in semicontinuous_specs:
            target_lb_j, target_ub_j = semicontinuous_specs[j_idx]
        else:
            target_ub_j = float(bounds[1, j_idx].item())
            # Strictly greater than tol so the projected variant is classified
            # as "active" by ``_classify_features_for_row`` (which treats
            # ``|x_j| <= tol`` as zero). Using ``tol`` exactly would round-trip
            # the variant back into the zero set and prevent loop convergence.
            target_lb_j = max(2 * tol, float(bounds[0, j_idx].item()))
        pin_j_to_zero = False

    # Build the fixed-pin dict: caller fixed_features + previously
    # committed zeros + (for ZERO) j_idx itself.
    fixed: Dict[int, float] = dict(fixed_features or {})
    for j in pinned_zero_indices or set():
        fixed[j] = 0.0
    if pin_j_to_zero:
        fixed[j_idx] = 0.0
    fixed_arg: Optional[Dict[int, float]] = fixed if fixed else None

    # Tighten bounds: per-action target on j_idx (active/activate) plus
    # bands for committed-active semi-continuous features.
    tightened = bounds.clone()
    if target_lb_j is not None:
        # target_lb_j and target_ub_j are set together by the per-kind
        # dispatch above — narrow the type for the assignment below.
        assert target_ub_j is not None
        tightened[0, j_idx] = target_lb_j
        tightened[1, j_idx] = target_ub_j
    if active_set and semicontinuous_specs:
        for i in (set(active_set) & semicontinuous_specs.keys()) - {j_idx}:
            lb_i, ub_i = semicontinuous_specs[i]
            tightened[0, i] = lb_i
            tightened[1, i] = ub_i

    # Build starting point: clip x_{j_idx} up to target_lb (active /
    # activate); apply fixed values.
    starting = x_i.clone()
    if target_lb_j is not None and float(starting[j_idx].item()) < target_lb_j:
        starting[j_idx] = target_lb_j
    for j, v in fixed.items():
        starting[j] = v

    has_linear = bool(inequality_constraints) or bool(equality_constraints)

    if not has_linear:
        variant = starting
        if per_step_local_reopt:
            refined = _local_optacqf(
                variant,
                tightened,
                fixed_arg,
                inequality_constraints,
                equality_constraints,
                acqf,
            )
            if refined is not None:
                variant = refined
        return variant, True

    try:
        projected = project_to_feasible_space_via_slsqp(
            X=starting.unsqueeze(0),
            bounds=tightened,
            inequality_constraints=inequality_constraints or None,
            equality_constraints=equality_constraints or None,
            fixed_features=fixed_arg,
        ).squeeze(0)
    except Exception:
        return starting, False

    # Safety net: SLSQP short-circuits when both lists are None,
    # and we trust the fixed_features convention rather than the
    # solver's idle path.
    projected = projected.clone()
    for j, v in fixed.items():
        projected[j] = v

    if per_step_local_reopt:
        refined = _local_optacqf(
            projected,
            tightened,
            fixed_arg,
            inequality_constraints,
            equality_constraints,
            acqf,
        )
        if refined is not None:
            return refined, True

    return projected, True


def _build_zero_variant(
    x_i: Tensor,
    j_idx: int,
    bounds: Tensor,
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
    equality_constraints: List[Tuple[Tensor, Tensor, float]],
    acqf: AcquisitionFunction,
    per_step_local_reopt: bool,
    pinned_zero_indices: Optional[Set[int]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    active_set: Optional[Set[int]] = None,
    semicontinuous_specs: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple[Tensor, bool]:
    """Thin wrapper around :func:`_build_variant` for the ZERO kind.

    Kept for backward compatibility with callers / tests that
    pre-date the unified builder.
    """
    return _build_variant(
        x_i=x_i,
        j_idx=j_idx,
        kind=ActionKind.ZERO,
        bounds=bounds,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        acqf=acqf,
        per_step_local_reopt=per_step_local_reopt,
        pinned_zero_indices=pinned_zero_indices,
        fixed_features=fixed_features,
        active_set=active_set,
        semicontinuous_specs=semicontinuous_specs,
    )


def _build_active_variant(
    x_i: Tensor,
    j_idx: int,
    lb_j: float,
    ub_j: float,
    bounds: Tensor,
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
    equality_constraints: List[Tuple[Tensor, Tensor, float]],
    acqf: AcquisitionFunction,
    per_step_local_reopt: bool,
    pinned_zero_indices: Optional[Set[int]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    active_set: Optional[Set[int]] = None,
    semicontinuous_specs: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple[Tensor, bool]:
    """Thin wrapper around :func:`_build_variant` for the ACTIVE kind.

    The legacy signature accepts ``lb_j`` / ``ub_j`` as positional
    args; we feed them into ``semicontinuous_specs`` (overriding any
    existing entry for ``j_idx``) so the unified dispatcher reads
    the same band.
    """
    specs: Dict[int, Tuple[float, float]] = dict(semicontinuous_specs or {})
    specs[j_idx] = (lb_j, ub_j)
    return _build_variant(
        x_i=x_i,
        j_idx=j_idx,
        kind=ActionKind.ACTIVE,
        bounds=bounds,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        acqf=acqf,
        per_step_local_reopt=per_step_local_reopt,
        pinned_zero_indices=pinned_zero_indices,
        fixed_features=fixed_features,
        active_set=active_set,
        semicontinuous_specs=specs,
    )


def _build_activate_variant(
    x_i: Tensor,
    j_idx: int,
    bounds: Tensor,
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
    equality_constraints: List[Tuple[Tensor, Tensor, float]],
    acqf: AcquisitionFunction,
    per_step_local_reopt: bool,
    pinned_zero_indices: Optional[Set[int]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    active_set: Optional[Set[int]] = None,
    semicontinuous_specs: Optional[Dict[int, Tuple[float, float]]] = None,
    tol: float = 1e-6,
) -> Tuple[Tensor, bool]:
    """Thin wrapper around :func:`_build_variant` for the ACTIVATE kind.

    See :func:`_build_variant` for the activation-target-band rules.
    """
    return _build_variant(
        x_i=x_i,
        j_idx=j_idx,
        kind=ActionKind.ACTIVATE,
        bounds=bounds,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        acqf=acqf,
        per_step_local_reopt=per_step_local_reopt,
        pinned_zero_indices=pinned_zero_indices,
        fixed_features=fixed_features,
        active_set=active_set,
        semicontinuous_specs=semicontinuous_specs,
        tol=tol,
    )


def _collect_actions(
    state: PruningState,
    ctx: PruningContext,
    fixed_keys: Set[int],
) -> List[Action]:
    """Build the admissible action set for ``state`` under ``ctx``.

    Replicates the legacy three-for-loop block of ``prune_nchoosek``:

    1. Compute per-constraint counts and the violation sets (max-count
       and min-count).
    2. Compute the per-kind eligibility sets, intersect with
       ``~fixed_keys``.
    3. For each eligible feature, apply the matching guard (min-count
       guard for ZERO, max-count guard for ACTIVATE) and, if the
       feature passes, build the variant via :func:`_build_variant`
       and append an :class:`Action` to the returned list.

    The action set may be empty even when constraints remain violated
    — for example, if the min-count guard filters every otherwise-
    eligible zero. The caller is responsible for raising
    :class:`PruningInfeasibleError` in that case; this helper only
    constructs the admissible set.
    """
    counts = _active_counts(
        state.x, ctx.nchoosek_constraints, ctx.features2idx, ctx.tol
    )
    violated = _violated_constraints(counts, ctx.nchoosek_constraints)
    min_count_violated = _min_count_violated_constraints(
        counts, ctx.nchoosek_constraints
    )

    eligible_zero = (
        _features_eligible_for_zero(
            state.frac_set,
            state.active_set,
            violated,
            ctx.nchoosek_constraints,
            ctx.features2idx,
        )
        - fixed_keys
    )
    eligible_activate = (
        _features_eligible_for_activate(
            state.zero_set,
            min_count_violated,
            ctx.nchoosek_constraints,
            ctx.features2idx,
        )
        - fixed_keys
    )

    actions: List[Action] = []

    for j in sorted(eligible_zero):
        if _zero_action_blocked_by_min_count(
            j, counts, ctx.nchoosek_constraints, ctx.features2idx
        ):
            continue
        variant, valid = _build_variant(
            x_i=state.x,
            j_idx=j,
            kind=ActionKind.ZERO,
            bounds=ctx.bounds,
            inequality_constraints=ctx.inequality_constraints,
            equality_constraints=ctx.equality_constraints,
            acqf=ctx.acqf,
            per_step_local_reopt=ctx.per_step_local_reopt,
            pinned_zero_indices=state.zero_set,
            fixed_features=ctx.fixed_features,
            active_set=state.active_set,
            semicontinuous_specs=ctx.semicontinuous_specs,
            tol=ctx.tol,
        )
        actions.append(Action(j=j, kind=ActionKind.ZERO, variant=variant, valid=valid))

    for j in sorted(state.frac_set):
        if j not in ctx.semicontinuous_specs:
            continue
        variant, valid = _build_variant(
            x_i=state.x,
            j_idx=j,
            kind=ActionKind.ACTIVE,
            bounds=ctx.bounds,
            inequality_constraints=ctx.inequality_constraints,
            equality_constraints=ctx.equality_constraints,
            acqf=ctx.acqf,
            per_step_local_reopt=ctx.per_step_local_reopt,
            pinned_zero_indices=state.zero_set,
            fixed_features=ctx.fixed_features,
            active_set=state.active_set,
            semicontinuous_specs=ctx.semicontinuous_specs,
            tol=ctx.tol,
        )
        actions.append(
            Action(j=j, kind=ActionKind.ACTIVE, variant=variant, valid=valid)
        )

    for j in sorted(eligible_activate):
        if _activate_action_blocked_by_max_count(
            j, counts, ctx.nchoosek_constraints, ctx.features2idx
        ):
            continue
        variant, valid = _build_variant(
            x_i=state.x,
            j_idx=j,
            kind=ActionKind.ACTIVATE,
            bounds=ctx.bounds,
            inequality_constraints=ctx.inequality_constraints,
            equality_constraints=ctx.equality_constraints,
            acqf=ctx.acqf,
            per_step_local_reopt=ctx.per_step_local_reopt,
            pinned_zero_indices=state.zero_set - {j},
            fixed_features=ctx.fixed_features,
            active_set=state.active_set,
            semicontinuous_specs=ctx.semicontinuous_specs,
            tol=ctx.tol,
        )
        actions.append(
            Action(j=j, kind=ActionKind.ACTIVATE, variant=variant, valid=valid)
        )

    return actions


def _final_local_reopt(
    x: Tensor,
    zero_set: Set[int],
    active_set: Set[int],
    semicontinuous_specs: Dict[int, Tuple[float, float]],
    bounds: Tensor,
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
    equality_constraints: List[Tuple[Tensor, Tensor, float]],
    acqf: AcquisitionFunction,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Tensor:
    """Run a single local ``optimize_acqf`` with the loop's decisions
    frozen.

    Zeroed features are pinned via ``fixed_features``; semi-continuous
    features in the active set get bounds tightened to
    ``[lb_j, ub_j]``. Caller-supplied ``fixed_features`` are merged in
    so they remain pinned during the clean-up. All other coordinates
    keep their global bounds. Falls back to ``x`` on optimiser failure.
    """
    tightened = bounds.clone()
    for j in active_set:
        if j in semicontinuous_specs:
            lb_j, ub_j = semicontinuous_specs[j]
            tightened[0, j] = lb_j
            tightened[1, j] = ub_j

    fixed: Dict[int, float] = dict(fixed_features or {})
    for j in zero_set:
        fixed[j] = 0.0

    refined = _local_optacqf(
        x,
        tightened,
        fixed if fixed else None,
        inequality_constraints,
        equality_constraints,
        acqf,
    )
    return refined if refined is not None else x


# ---------------------------------------------------------------------------
# AF evaluation
# ---------------------------------------------------------------------------


def _evaluate_variants_with_prefix(
    X_prefix: Tensor,
    variants: Tensor,
    acqf: AcquisitionFunction,
) -> Tensor:
    """Batched AF evaluation. ``X_prefix`` is ``(i, d)`` (possibly
    empty); ``variants`` is ``(R, d)``; returns ``(R,)``.
    """
    R = variants.shape[0]
    if X_prefix.shape[0] > 0:
        prefix = X_prefix.unsqueeze(0).expand(R, -1, -1)
        eval_X = torch.cat([prefix, variants.unsqueeze(1)], dim=1)
    else:
        eval_X = variants.unsqueeze(1)
    with torch.no_grad():
        return acqf(eval_X)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _prune_single_candidate(
    x_i: Tensor,
    X_prefix: Tensor,
    ctx: PruningContext,
    fixed_keys: Set[int],
) -> Tuple[Tensor, PruningState]:
    """Prune one candidate row to NChooseK + semi-continuity feasibility.

    The greedy inner loop: at every iteration, build the admissible
    action set (zero / active / activate variants), evaluate the
    acquisition function at each variant, and commit the action with
    the smallest AF reduction. Loop terminates when no fractional
    feature remains and every NChooseK constraint is satisfied.

    ``X_prefix`` is the joint context for q-batch AF conditioning: it
    is a snapshot of already-pruned earlier candidates, taken at outer-
    iteration entry, and is never modified by this function. Both the
    variant AF evaluation (via :func:`_evaluate_variants_with_prefix`)
    and the dense-AF baseline are computed conditional on
    ``X_prefix``, so candidate ``i+1``'s decisions account for the AF
    reductions caused by candidate ``i``'s pruned form (not its dense
    form).

    Returns ``(x_pruned, final_state)``: the pruned candidate tensor
    and the final :class:`PruningState`. The caller can use the
    state's ``zero_set`` / ``active_set`` to drive the optional final
    local re-optimisation. ``x_i`` itself is not mutated.

    Raises:
        PruningInfeasibleError: if the greedy loop cannot satisfy all
            constraints (typically because the per-step min/max-count
            guards empty the action set, or because the iteration cap
            ``2 × n_features`` is exceeded).
    """
    zero_set, frac_set, active_set = _classify_features_for_row(
        x_i, ctx.semicontinuous_specs, ctx.tol
    )
    # Caller-fixed features are never proposed as actions.
    frac_set -= fixed_keys
    active_set -= fixed_keys
    # Caller-fixed features whose value is exactly zero get tracked
    # as part of zero_set for the action eligibility filter, but are
    # not emitted by the loop.
    for j, v in (ctx.fixed_features or {}).items():
        if abs(v) <= ctx.tol:
            zero_set.add(j)

    state = PruningState(
        x=x_i.clone(),
        zero_set=zero_set,
        frac_set=frac_set,
        active_set=active_set,
    )

    # Activate is non-monotone (a feature can flip
    # zero → active → zero across iterations if AF preferences shift),
    # so cap iterations defensively. 2*d is well above the
    # AF-driven greedy's typical convergence (≤ d).
    max_inner_iters = 2 * x_i.shape[0]
    inner_iter = 0

    while True:
        if not state.frac_set and _is_nchoosek_fulfilled(
            state.x,
            ctx.nchoosek_constraints,
            ctx.features2idx,
            ctx.tol,
        ):
            break
        if inner_iter >= max_inner_iters:
            raise PruningInfeasibleError(
                f"Greedy pruning exceeded the per-candidate iteration "
                f"cap of {max_inner_iters} (2 × n_features) without "
                f"reaching feasibility. This usually indicates an "
                f"oscillation between zero and activate actions on "
                f"the same feature. Inspect the acquisition function "
                f"for non-determinism or extreme non-convexity."
            )
        inner_iter += 1

        actions = _collect_actions(state, ctx, fixed_keys)

        if not actions:
            raise PruningInfeasibleError(
                "Greedy pruning could not satisfy NChooseK + "
                "semi-continuity constraints: action set empty "
                "while constraints still violated."
            )

        variants = torch.stack([a.variant for a in actions])
        af_values = _evaluate_variants_with_prefix(X_prefix, variants, ctx.acqf)
        valid_flags = torch.tensor(
            [a.valid for a in actions],
            device=af_values.device,
        )
        af_values = torch.where(
            valid_flags,
            af_values,
            torch.full_like(af_values, float("-inf")),
        )

        # BONSAI greedy rule in its minimal form: argmax of joint AF
        # over variants (with -inf masking invalid ones), tie-break
        # by ZERO < ACTIVE < ACTIVATE, then by smaller j_idx. The
        # paper writes this as `argmin (α(x_dense) − α(x_variant))`;
        # the two forms are equivalent because `α(x_dense)` is the
        # same scalar across all variants within an iteration, so
        # the subtraction is a constant offset that cancels in the
        # argmin. Cumulative AF reduction across iterations also
        # telescopes (`α(x_root) − α(x_terminal)`), so no stable
        # reference is needed for downstream comparisons either.
        best_k = min(
            range(len(actions)),
            key=lambda k: (
                -float(af_values[k].item()),
                actions[k].kind.value,
                actions[k].j,
            ),
        )

        chosen = actions[best_k]
        state.x = chosen.variant
        state.commit(chosen)

    return state.x, state


def prune_nchoosek(
    X: Tensor,
    acqf: AcquisitionFunction,
    nchoosek_constraints: Sequence[NChooseKConstraint],
    features2idx: Dict[str, Tuple[int, ...]],
    bounds: Tensor,
    inequality_constraints: List[Tuple[Tensor, Tensor, float]],
    equality_constraints: List[Tuple[Tensor, Tensor, float]],
    semicontinuous_specs: Dict[int, Tuple[float, float]],
    *,
    fixed_features: Optional[Dict[int, float]] = None,
    per_step_local_reopt: bool = False,
    final_local_reopt: bool = True,
    tol: float = 1e-6,
) -> Tensor:
    """Greedy BONSAI pruning of a q-batch of acquisition-optimised
    candidates.

    For every candidate row ``X[i]``, iteratively commit the action
    whose acquisition reduction is smallest, until every NChooseK
    constraint is satisfied and no feature remains in the
    semi-continuous gap. Three action kinds:

    - ``zero(j)``: zero a currently active or fractional feature.
    - ``active(j)``: snap a fractional semi-continuous feature into
      its ``[lb_j, ub_j]`` band.
    - ``activate(j)``: bring a currently-zero feature into a positive
      value, used when some NChooseK constraint has
      ``a_c < min_count_c``. Targets ``[lb_j, ub_j]`` for
      semi-continuous features; ``[tol, ub_j]`` otherwise.

    On constraints with ``none_also_valid=True``, the loop activates
    when ``0 < count < min_count``; it does *not* attempt to zero
    down to ``count=0`` (the per-step zero guard blocks intermediate
    sub-min states). ``count=0`` is reached only if the AF maximiser
    places the candidate there to begin with.

    Later candidates condition on already-pruned earlier ones via the
    prefixed AF evaluation.

    Args:
        X: ``(q, d)`` candidate tensor.
        acqf: BoTorch acquisition function. Called as ``acqf(X)`` with
            ``(b, q', d)`` shaped input.
        nchoosek_constraints: NChooseK constraints (filtered list,
            i.e. no other constraint types).
        features2idx: Mapping from feature key to a tuple of tensor
            column indices (one entry per column, length 1 for plain
            ``ContinuousInput``).
        bounds: ``(2, d)`` tensor of lower and upper bounds.
        inequality_constraints: BoTorch-style ``(indices, coefficients,
            rhs)`` triples for ``A x >= b``. May be empty.
        equality_constraints: same format, for ``A x = b``. May be empty.
        semicontinuous_specs: ``{j_idx: (lb_j, ub_j)}`` for every
            tensor column whose feature is ``allow_zero=True`` with
            ``lb_j > 0``. May be empty when no semi-continuous
            features are in the domain.
        fixed_features: ``{j_idx: value}`` for caller-supplied features
            that must remain at the given value throughout pruning
            (e.g., inputs whose bounds collapse to a single value).
            These features are never proposed as zero or active
            actions, and they are forwarded to both the QP projection
            and ``optimize_acqf`` so they cannot drift. May be empty
            or ``None``.
        per_step_local_reopt: When ``True``, every variant
            (zero or active) is locally re-optimised via
            ``optimize_acqf`` after the QP projection. This is more
            accurate but doubles or triples the per-iteration cost.
            When ``False``, the projection result is used directly.
        final_local_reopt: When ``True``, after the greedy loop
            completes a single ``optimize_acqf`` clean-up pass is run
            with zeroed features fixed and active semi-continuous
            features bounded to their active band.
        tol: Tolerance used for "is this feature zero?" classification
            and for the fulfilment check.

    Returns:
        ``(q, d)`` tensor of pruned candidates. The tensor is a
        clone — the input ``X`` is not mutated.

    Raises:
        PruningInfeasibleError: if the greedy loop cannot satisfy all
            constraints (typically because the ``min_count`` guard
            empties the action set before the ``max_count`` constraints
            are met).
    """
    # Defensive guard: NChooseK features must each map to a single tensor
    # column. The NChooseKConstraint validator already restricts to
    # ContinuousInput, but the optimizer's input_preprocessing_specs could
    # in principle multi-encode an ill-typed feature; fail loudly if so.
    for c in nchoosek_constraints:
        for feat_key in c.features:
            cols = features2idx.get(feat_key, ())
            if len(cols) != 1:
                raise NotImplementedError(
                    f"Pruning requires every NChooseK feature to map to a "
                    f"single tensor column. Feature {feat_key!r} maps to "
                    f"{cols} via features2idx — this typically means it "
                    f"was one-hot or otherwise multi-column encoded, which "
                    f"is out of scope for the BONSAI pruning module."
                )

    X = X.clone()
    q = X.shape[0]
    if q == 0:
        return X

    fixed_keys: Set[int] = set((fixed_features or {}).keys())
    ctx = PruningContext(
        bounds=bounds,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        acqf=acqf,
        semicontinuous_specs=semicontinuous_specs,
        fixed_features=fixed_features,
        nchoosek_constraints=nchoosek_constraints,
        features2idx=features2idx,
        tol=tol,
        per_step_local_reopt=per_step_local_reopt,
    )

    for i in range(q):
        # Snapshot the prefix so `_prune_single_candidate` cannot
        # accidentally observe (or mutate) later rows of X. This is
        # also the invariant the future beam/B&B refactor needs:
        # multiple in-flight states must share a single immutable
        # prefix per outer iteration.
        x_pruned, final_state = _prune_single_candidate(
            X[i], X[:i].clone(), ctx, fixed_keys
        )
        X[i] = x_pruned
        if final_local_reopt:
            X[i] = _final_local_reopt(
                X[i],
                final_state.zero_set - fixed_keys,
                final_state.active_set,
                semicontinuous_specs,
                bounds,
                inequality_constraints,
                equality_constraints,
                acqf,
                fixed_features=fixed_features,
            )

    return X
