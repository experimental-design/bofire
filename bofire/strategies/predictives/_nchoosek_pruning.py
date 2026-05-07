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


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------
#
# TODO: tighten bounds for *all* committed active semi-continuous features
# inside _build_zero_variant and _build_active_variant, not just the
# target column. Currently each variant builder only tightens the bounds
# of the feature it acts on:
#
#     # _build_active_variant — tightens only j_idx
#     tightened[0, j_idx] = lb_j
#     tightened[1, j_idx] = ub_j
#
#     # _build_zero_variant — tightens nothing, just sets fixed_features[j_idx]=0
#
# All other previously-committed active semi-continuous features keep
# the relaxed bounds (lb=0) that the AF maximiser used. So when iter k+1
# runs optimize_acqf for action active(j_{k+1}) (or zero(j_{k+1})), the
# optimiser is free to push x_{j_k} (committed active in iter k) back
# into the gap (0, lb_{j_k}) to satisfy the mixture or other shared
# constraints. State tracker says x_{j_k} is active; tensor disagrees.
# The drift is only corrected by `_final_local_reopt`, which tightens
# bounds on every active semi-continuous feature simultaneously.
#
# The principled fix is symmetric in both builders:
#
#     for i in (active_set & semicontinuous_specs.keys()) - {j_idx}:
#         tightened[0, i] = lb_i
#         tightened[1, i] = ub_i
#
# The `- {j_idx}` exclusion is essential: it preserves the ability to
# *deactivate* (zero) a previously-committed active feature. Without
# the exclusion, zero(j) would set bounds[0, j] = lb_j AND
# fixed_features[j] = 0 simultaneously — infeasible. With the
# exclusion, the feature being acted on is exempt from the tightening
# (because the action itself defines its target value/bounds), while
# every other already-committed active stays locked into [lb, ub].
#
# After this fix:
#   - per-step reopt is internally consistent, no drift across
#     iterations,
#   - `_final_local_reopt` reduces to its named role (end-of-loop
#     polish), not a feasibility rescue,
#   - `final_local_reopt` could safely default to False on every
#     scenario, semi-continuous included.
#
# This requires `_build_*_variant` to receive the loop's current
# `active_set` (today they only get `pinned_zero_indices`). Plumbing
# is small. Worth bundling with the activate-zero work for stage 3
# (no semi-continuous benchmarks in stage 1/2 exercise this code path,
# so the bug is currently latent — `final_local_reopt=True` papers
# over it for the only domain shape in which it would surface today).
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
) -> Tuple[Tensor, bool]:
    """Construct the candidate variant where ``x_j`` is forced to zero.

    ``pinned_zero_indices`` lists tensor columns that have been
    committed to zero in earlier greedy iterations. They are passed
    as additional ``fixed_features`` to the QP projection and (when
    enabled) the local re-optimisation, preventing the linear-
    constraint redistribution from resurrecting previously-zeroed
    features.

    ``fixed_features`` is the caller-supplied mapping of features
    that must remain at fixed values throughout pruning (e.g.,
    inputs whose bounds collapse to a single value). They are merged
    into the projection's and local re-optimiser's ``fixed_features``
    so SLSQP and ``optimize_acqf`` never move them.

    Returns ``(variant, valid)``:

    - ``valid=True`` if the variant is a feasible point (linear-
      constraint compatible);
    - ``valid=False`` if the QP projection failed (mutually
      infeasible bounds + linear constraints). The variant is still
      returned with ``x[j_idx] = 0`` so the caller has a well-formed
      tensor; the caller must replace its AF value with ``-inf`` so
      it is never selected.
    """
    has_linear = bool(inequality_constraints) or bool(equality_constraints)
    fixed: Dict[int, float] = dict(fixed_features or {})
    for j in pinned_zero_indices or set():
        fixed[j] = 0.0
    fixed[j_idx] = 0.0

    if not has_linear:
        variant = x_i.clone()
        for j, v in fixed.items():
            variant[j] = v
        if per_step_local_reopt:
            refined = _local_optacqf(
                variant,
                bounds,
                fixed,
                inequality_constraints,
                equality_constraints,
                acqf,
            )
            if refined is not None:
                variant = refined
        return variant, True

    try:
        projected = project_to_feasible_space_via_slsqp(
            X=x_i.unsqueeze(0),
            bounds=bounds,
            inequality_constraints=inequality_constraints or None,
            equality_constraints=equality_constraints or None,
            fixed_features=fixed,
        ).squeeze(0)
    except Exception:
        fallback = x_i.clone()
        for j, v in fixed.items():
            fallback[j] = v
        return fallback, False

    # Safety net: SLSQP short-circuits when both lists are None,
    # and we trust the fixed_features convention rather than the
    # solver's idle path.
    projected = projected.clone()
    for j, v in fixed.items():
        projected[j] = v

    if per_step_local_reopt:
        refined = _local_optacqf(
            projected,
            bounds,
            fixed,
            inequality_constraints,
            equality_constraints,
            acqf,
        )
        if refined is not None:
            return refined, True

    return projected, True


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
) -> Tuple[Tensor, bool]:
    """Construct the variant where ``x_j`` is snapped into ``[lb_j, ub_j]``.

    The local bounds are tightened on column ``j_idx`` so the
    optimiser is constrained to the active branch of the
    semi-continuous feature. Other columns retain the global bounds.

    ``pinned_zero_indices`` keeps previously-zeroed coordinates pinned
    via ``fixed_features`` so the projection cannot resurrect them.
    ``fixed_features`` is the caller-supplied mapping of features
    that must remain at their fixed values throughout pruning.
    """
    tightened = bounds.clone()
    tightened[0, j_idx] = lb_j
    tightened[1, j_idx] = ub_j

    starting = x_i.clone()
    if float(starting[j_idx].item()) < lb_j:
        starting[j_idx] = lb_j

    fixed: Dict[int, float] = dict(fixed_features or {})
    for j in pinned_zero_indices or set():
        fixed[j] = 0.0
    for j, v in fixed.items():
        starting[j] = v
    fixed_arg: Optional[Dict[int, float]] = fixed if fixed else None

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


def _af_reduction(
    dense_af: Tensor,
    variant_af: Tensor,
) -> Tensor:
    """Absolute BONSAI AF reduction: ``g_j = α(x_dense) − α(x_pruned_j)``.

    Smaller is better. ``argmin`` picks the variant whose pruning costs the
    least acquisition value — the BONSAI greedy rule from
    https://arxiv.org/abs/2602.07144.

    Equivalent to ``argmax variant_af`` since ``dense_af`` is the same for
    every variant in a single iteration; we keep the explicit subtraction
    to mirror the paper's notation. We deliberately do not normalise by
    the dense incremental because (a) the paper only normalises in the
    termination criterion (the ρ threshold), not in selection, and (b) we
    terminate by NChooseK satisfaction rather than ρ, so the
    normalisation has no semantic role here. The relative form is also
    ill-defined when ``dense_af ≤ base_af`` (qLogEI on a data-starved
    candidate, etc.); the absolute form has no such pathology.
    """
    return dense_af - variant_af


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


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
    (zero a feature, or snap a fractional feature into its active
    band) whose acquisition reduction is smallest, until every
    NChooseK constraint is satisfied and no feature remains in the
    semi-continuous gap. Later candidates condition on already-pruned
    earlier ones via the prefixed AF evaluation.

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

    for i in range(q):
        zero_set, frac_set, active_set = _classify_features_for_row(
            X[i],
            semicontinuous_specs,
            tol,
        )
        # Caller-fixed features are never proposed as actions.
        frac_set -= fixed_keys
        active_set -= fixed_keys
        # Caller-fixed features whose value is exactly zero get
        # tracked as part of zero_set for the action eligibility
        # filter, but are not emitted by the loop.
        for j, v in (fixed_features or {}).items():
            if abs(v) <= tol:
                zero_set.add(j)

        while True:
            if not frac_set and _is_nchoosek_fulfilled(
                X[i],
                nchoosek_constraints,
                features2idx,
                tol,
            ):
                break

            counts = _active_counts(
                X[i],
                nchoosek_constraints,
                features2idx,
                tol,
            )
            violated = _violated_constraints(counts, nchoosek_constraints)
            eligible = _features_eligible_for_zero(
                frac_set,
                active_set,
                violated,
                nchoosek_constraints,
                features2idx,
            )

            # Caller-fixed features are never moved by the loop.
            eligible = eligible - fixed_keys

            actions: List[Tuple[int, str, Tensor, bool]] = []
            for j in sorted(eligible):
                if _zero_action_blocked_by_min_count(
                    j,
                    counts,
                    nchoosek_constraints,
                    features2idx,
                ):
                    continue
                variant, valid = _build_zero_variant(
                    X[i],
                    j,
                    bounds,
                    inequality_constraints,
                    equality_constraints,
                    acqf,
                    per_step_local_reopt,
                    pinned_zero_indices=zero_set,
                    fixed_features=fixed_features,
                )
                actions.append((j, "zero", variant, valid))

            for j in sorted(frac_set):
                if j not in semicontinuous_specs:
                    continue
                lb_j, ub_j = semicontinuous_specs[j]
                variant, valid = _build_active_variant(
                    X[i],
                    j,
                    lb_j,
                    ub_j,
                    bounds,
                    inequality_constraints,
                    equality_constraints,
                    acqf,
                    per_step_local_reopt,
                    pinned_zero_indices=zero_set,
                    fixed_features=fixed_features,
                )
                actions.append((j, "active", variant, valid))

            if not actions:
                raise PruningInfeasibleError(
                    "Greedy pruning could not satisfy NChooseK + "
                    "semi-continuity constraints: action set empty "
                    "while constraints still violated."
                )

            variants = torch.stack([a[2] for a in actions])
            af_values = _evaluate_variants_with_prefix(X[:i], variants, acqf)
            valid_flags = torch.tensor(
                [a[3] for a in actions],
                device=af_values.device,
            )
            af_values = torch.where(
                valid_flags,
                af_values,
                torch.full_like(af_values, float("-inf")),
            )

            dense_af = acqf(X[: i + 1].unsqueeze(0)).detach()
            af_red = _af_reduction(dense_af, af_values)

            # Stable tie-break: smallest af_reduction first; on ties,
            # prefer "zero" over "active", then smaller j_idx.
            best_k = 0
            best_key = (
                float(af_red[0].item()),
                0 if actions[0][1] == "zero" else 1,
                actions[0][0],
            )
            for k in range(1, len(actions)):
                key = (
                    float(af_red[k].item()),
                    0 if actions[k][1] == "zero" else 1,
                    actions[k][0],
                )
                if key < best_key:
                    best_key = key
                    best_k = k

            j_pick, kind_pick, variant_pick, _ = actions[best_k]
            X[i] = variant_pick

            # Update per-candidate state.
            if kind_pick == "zero":
                frac_set.discard(j_pick)
                active_set.discard(j_pick)
                zero_set.add(j_pick)
            else:
                frac_set.discard(j_pick)
                active_set.add(j_pick)

        if final_local_reopt:
            X[i] = _final_local_reopt(
                X[i],
                zero_set - fixed_keys,
                active_set,
                semicontinuous_specs,
                bounds,
                inequality_constraints,
                equality_constraints,
                acqf,
                fixed_features=fixed_features,
            )

    return X
