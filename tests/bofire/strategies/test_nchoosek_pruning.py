"""Unit tests for the standalone NChooseK pruning module.

Phase 1 verification of `bofire/strategies/predictives/_nchoosek_pruning.py`.
"""

from typing import Any, Dict, List, Tuple, cast

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.testing import MockAcquisitionFunction
from torch import Tensor

from bofire.data_models.constraints.api import (
    InterpointConstraint,
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearConstraint,
    ProductConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.encodings.api import OneHotEncoding
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.strategies.predictives import _nchoosek_pruning as ncp
from bofire.strategies.predictives._nchoosek_pruning import (
    Action,
    ActionKind,
    PruningInfeasibleError,
    PruningState,
    _activate_action_blocked_by_max_count,
    _active_counts,
    _build_variant,
    _classify_features_for_row,
    _features_eligible_for_activate,
    _features_eligible_for_zero,
    _is_nchoosek_fulfilled,
    _max_count_violated_constraints,
    _min_count_violated_constraints,
    _zero_action_blocked_by_min_count,
    has_nchoosek_linear_overlap,
    has_semicontinuous_features,
    is_nchoosek_pruning_applicable,
    is_pruning_applicable,
    prune_nchoosek,
)
from bofire.utils.torch_tools import (
    get_linear_constraints,
    get_torch_bounds_from_domain,
    tkwargs,
)


# ---------------------------------------------------------------------------
# Test acquisition functions
# ---------------------------------------------------------------------------


class WeightedSumAcqf:
    """Deterministic AF over the q-batch.

    For input ``X`` of shape ``(b, q, d)``, returns ``(b,)`` where each
    value is the maximum over the q-batch of ``(X * weights).sum(-1)``.
    Useful for tests that need to assert *which* feature was zeroed.
    """

    def __init__(self, weights: Tensor):
        self.weights = weights
        self.model = None
        self.X_pending = None

    def __call__(self, X: Tensor) -> Tensor:
        return (X * self.weights).sum(-1).max(dim=-1).values

    def set_X_pending(self, X_pending=None):
        self.X_pending = X_pending


class ConstantAcqf:
    """AF returning a constant for every input — exercises the case
    where every variant has the same absolute AF and the tie-break runs.
    """

    def __init__(self, value: float = 1.0):
        self.value = value
        self.model = None
        self.X_pending = None

    def __call__(self, X: Tensor) -> Tensor:
        b = X.shape[0]
        return torch.full((b,), self.value, **tkwargs)

    def set_X_pending(self, X_pending=None):
        self.X_pending = X_pending


class RecordingCallAF:
    """Wraps a base AF and captures every ``__call__`` input tensor.

    Used by tests that verify what gets passed to the AF during the
    pruning loop (e.g. q-batch prefix conditioning).
    """

    def __init__(self, base: Any):
        self.base = base
        self.calls: List[Tensor] = []

    def __call__(self, X: Tensor) -> Tensor:
        self.calls.append(X.detach().clone())
        return self.base(X)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)


def _make_recording_pending_af(base: Any) -> Tuple[Any, List[Any]]:
    """Wrap ``base`` so every ``set_X_pending`` call is captured.

    Returns ``(wrapped, calls)`` where ``calls`` is a list of
    ``("set", clone)`` tuples in invocation order.
    """
    calls: List[Any] = []

    class RecordingPendingAF:
        def __init__(self) -> None:
            self.base = base
            self.X_pending: Any = None  # exposed for the decorator path

        def set_X_pending(self, value: Any) -> None:
            clone = None if value is None else value.detach().clone()
            calls.append(("set", clone))
            self.X_pending = clone

        def __call__(self, X: Tensor) -> Tensor:
            return self.base(X)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.base, name)

    return RecordingPendingAF(), calls


# ---------------------------------------------------------------------------
# Domain → algorithm-input helper
# ---------------------------------------------------------------------------


def _inputs_from_domain(domain: Domain) -> Dict[str, Any]:
    """Build the kwargs for ``prune_nchoosek`` from a Domain.

    Mirrors what ``BotorchOptimizer`` will do in Phase 2.
    """
    specs: Dict[str, Any] = {}  # pure continuous => no encoding
    features2idx, _ = domain.inputs._get_transform_info(specs)

    bounds = get_torch_bounds_from_domain(domain, specs)

    inequality_constraints = get_linear_constraints(
        domain, constraint=LinearInequalityConstraint
    )
    equality_constraints = get_linear_constraints(
        domain, constraint=LinearEqualityConstraint
    )

    nchoosek_constraints = list(domain.constraints.get(NChooseKConstraint))

    semicontinuous_specs: Dict[int, Tuple[float, float]] = {}
    for feat in domain.inputs.get(ContinuousInput):
        assert isinstance(feat, ContinuousInput)
        if feat.allow_zero and feat.bounds[0] > 0:
            for col in features2idx[feat.key]:
                semicontinuous_specs[col] = (
                    float(feat.bounds[0]),
                    float(feat.bounds[1]),
                )
                # Relax the lower bound for AF-optimisation / pruning:
                # the convex relaxation feasible region is [0, ub] for
                # semi-continuous features. The active variant inside
                # prune_nchoosek tightens this back to [lb, ub] per
                # feature when needed.
                bounds[0, col] = 0.0

    return {
        "nchoosek_constraints": nchoosek_constraints,
        "features2idx": features2idx,
        "bounds": bounds,
        "inequality_constraints": inequality_constraints,
        "equality_constraints": equality_constraints,
        "semicontinuous_specs": semicontinuous_specs,
    }


def _row_to_tensor(row: List[float]) -> Tensor:
    return torch.tensor(row, **tkwargs)


def _stack_to_tensor(rows: List[List[float]]) -> Tensor:
    return torch.tensor(rows, **tkwargs)


def _make_simple_domain(
    *,
    n_features: int = 4,
    allow_zero: bool = False,
    lb: float = 0.0,
    ub: float = 1.0,
    constraints: List = None,
) -> Domain:
    if constraints is None:
        constraints = []
    inputs = [
        ContinuousInput(
            key=f"x{i + 1}",
            bounds=(lb, ub),
            allow_zero=allow_zero,
        )
        for i in range(n_features)
    ]
    outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
    return Domain.from_lists(inputs=inputs, outputs=outputs, constraints=constraints)


def _default_nchoosek_setup(
    *,
    n_features: int = 4,
    min_count: int = 1,
    max_count: int = 2,
    none_also_valid: bool = False,
) -> Tuple[Domain, Dict[str, Any]]:
    """Canonical (domain, inp) setup used by most pruning tests:
    `n_features` continuous inputs over `[0, 1]` plus a single NChooseK
    over all of them. Returns the domain plus the kwargs dict
    `_inputs_from_domain` produces for `prune_nchoosek`.
    """
    feature_keys = [f"x{i + 1}" for i in range(n_features)]
    domain = _make_simple_domain(
        n_features=n_features,
        constraints=[
            NChooseKConstraint(
                features=feature_keys,
                min_count=min_count,
                max_count=max_count,
                none_also_valid=none_also_valid,
            ),
        ],
    )
    return domain, _inputs_from_domain(domain)


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def _basic_constraint(self, **kw) -> NChooseKConstraint:
        defaults = {
            "features": ["x1", "x2", "x3"],
            "min_count": 1,
            "max_count": 2,
            "none_also_valid": False,
        }
        defaults.update(kw)
        return NChooseKConstraint(**defaults)  # type: ignore

    def _f2i_continuous(self, n: int) -> Dict[str, Tuple[int, ...]]:
        return {f"x{i + 1}": (i,) for i in range(n)}

    # ----- _classify_features_for_row -----

    def test_classify_no_semicontinuous_partitions_zero_and_active(self):
        x = _row_to_tensor([0.0, 0.5, 1e-9, 0.7])
        zero, frac, active = _classify_features_for_row(x, {}, tol=1e-6)
        assert zero == {0, 2}
        assert frac == set()
        assert active == {1, 3}

    def test_classify_semicontinuous_value_in_gap_is_fractional(self):
        x = _row_to_tensor([0.0, 0.1, 0.5, 0.0])
        semi = {1: (0.2, 1.0), 2: (0.2, 1.0)}
        zero, frac, active = _classify_features_for_row(x, semi, tol=1e-6)
        assert zero == {0, 3}
        assert frac == {1}
        assert active == {2}

    def test_classify_value_at_lb_is_active_not_fractional(self):
        x = _row_to_tensor([0.2, 0.0, 0.5, 0.0])
        semi = {0: (0.2, 1.0)}
        zero, frac, active = _classify_features_for_row(x, semi, tol=1e-6)
        assert frac == set()
        assert active == {0, 2}
        assert zero == {1, 3}

    # ----- _is_nchoosek_fulfilled -----

    def test_fulfilled_max_count_violated(self):
        c = self._basic_constraint(max_count=2)
        x = _row_to_tensor([0.5, 0.5, 0.5])
        assert (
            _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3), tol=1e-6) is False
        )

    def test_fulfilled_min_count_violated(self):
        c = self._basic_constraint(min_count=2, max_count=3)
        x = _row_to_tensor([0.5, 0.0, 0.0])
        assert (
            _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3), tol=1e-6) is False
        )

    def test_fulfilled_none_also_valid_zero_count(self):
        c = self._basic_constraint(min_count=1, max_count=2, none_also_valid=True)
        x = _row_to_tensor([0.0, 0.0, 0.0])
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3), tol=1e-6) is True

    def test_fulfilled_none_also_valid_does_not_allow_low_nonzero(self):
        c = self._basic_constraint(min_count=2, max_count=3, none_also_valid=True)
        x = _row_to_tensor([0.5, 0.0, 0.0])  # one active, below min_count
        assert (
            _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3), tol=1e-6) is False
        )

    def test_fulfilled_tol_classifies_small_values_as_zero(self):
        c = self._basic_constraint(min_count=1, max_count=2)
        x = _row_to_tensor([1e-9, 1e-9, 0.5])
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3), tol=1e-6) is True

    def test_fulfilled_within_bounds(self):
        c = self._basic_constraint(min_count=1, max_count=2)
        x = _row_to_tensor([0.5, 0.5, 0.0])
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3), tol=1e-6) is True

    # ----- _active_counts -----

    def test_active_counts_per_constraint(self):
        c1 = self._basic_constraint(features=["x1", "x2"], max_count=1)
        c2 = self._basic_constraint(features=["x2", "x3", "x4"], max_count=2)
        x = _row_to_tensor([0.5, 0.5, 0.5, 0.0])
        f2i = self._f2i_continuous(4)
        counts = _active_counts(x, [c1, c2], f2i, tol=1e-6)
        assert counts == {0: 2, 1: 2}

    # ----- _max_count_violated_constraints -----

    def test_violated_constraints_only_max_violations(self):
        c1 = self._basic_constraint(features=["x1", "x2"], max_count=1, min_count=0)
        c2 = self._basic_constraint(features=["x3", "x4"], max_count=2, min_count=2)
        counts = {0: 2, 1: 1}  # c1 violates max, c2 violates min (but excluded)
        violated = _max_count_violated_constraints(counts, [c1, c2])
        assert violated == {0}

    # ----- _zero_action_blocked_by_min_count -----

    def test_min_count_guard_blocks_zero(self):
        c = self._basic_constraint(min_count=2, max_count=3, none_also_valid=False)
        f2i = self._f2i_continuous(3)
        counts = {0: 2}  # at the floor; zeroing any feature drops below
        assert _zero_action_blocked_by_min_count(0, counts, [c], f2i) is True

    def test_min_count_guard_allows_zero_above_floor(self):
        c = self._basic_constraint(min_count=2, max_count=3, none_also_valid=False)
        f2i = self._f2i_continuous(3)
        counts = {0: 3}
        assert _zero_action_blocked_by_min_count(0, counts, [c], f2i) is False

    def test_min_count_guard_none_also_valid_allows_drop_to_zero(self):
        c = self._basic_constraint(min_count=2, max_count=3, none_also_valid=True)
        f2i = self._f2i_continuous(3)
        counts = {0: 1}  # dropping to zero is allowed by none_also_valid
        assert _zero_action_blocked_by_min_count(0, counts, [c], f2i) is False

    def test_min_count_guard_none_also_valid_blocks_drop_to_intermediate(self):
        c = self._basic_constraint(min_count=2, max_count=3, none_also_valid=True)
        f2i = self._f2i_continuous(3)
        counts = {0: 2}  # dropping to 1 is blocked, 1 < min_count and not zero
        assert _zero_action_blocked_by_min_count(0, counts, [c], f2i) is True

    def test_min_count_guard_ignores_disjoint_constraints(self):
        # j_idx=0 (x1) is not in c2 — c2 should not affect the verdict
        c1 = self._basic_constraint(features=["x1", "x2"], min_count=0, max_count=2)
        c2 = self._basic_constraint(features=["x3", "x4"], min_count=2, max_count=2)
        f2i = self._f2i_continuous(4)
        counts = {0: 2, 1: 2}
        assert _zero_action_blocked_by_min_count(0, counts, [c1, c2], f2i) is False

    # ----- _features_eligible_for_zero -----

    def test_eligible_includes_all_fractional(self):
        c = self._basic_constraint()
        f2i = self._f2i_continuous(3)
        eligible = _features_eligible_for_zero(
            fractional={0},
            active={1},
            violated=set(),
            nchoosek_constraints=[c],
            features2idx=f2i,
        )
        assert eligible == {0}

    def test_eligible_includes_active_in_violated_constraint(self):
        c = self._basic_constraint(features=["x1", "x2"], max_count=1)
        f2i = self._f2i_continuous(3)
        eligible = _features_eligible_for_zero(
            fractional=set(),
            active={0, 1, 2},
            violated={0},
            nchoosek_constraints=[c],
            features2idx=f2i,
        )
        # x3 is not in c, so excluded; x1 and x2 are
        assert eligible == {0, 1}

    def test_eligible_excludes_active_outside_any_violated(self):
        c = self._basic_constraint(features=["x1", "x2"], max_count=2)
        f2i = self._f2i_continuous(3)
        eligible = _features_eligible_for_zero(
            fractional=set(),
            active={0, 1, 2},
            violated=set(),
            nchoosek_constraints=[c],
            features2idx=f2i,
        )
        assert eligible == set()


# ---------------------------------------------------------------------------
# Variant construction tests
# ---------------------------------------------------------------------------


class TestVariantConstruction:
    def _bounds(self, n: int = 4) -> Tensor:
        return torch.tensor([[0.0] * n, [1.0] * n], **tkwargs)

    def _acqf(self) -> Any:
        return cast(AcquisitionFunction, MockAcquisitionFunction())

    def test_zero_variant_no_constraints_zeroes_in_place(self):
        x = _row_to_tensor([0.5, 0.5, 0.5, 0.5])
        variant, valid = _build_variant(
            x,
            j_idx=2,
            kind=ActionKind.ZERO,
            bounds=self._bounds(),
            inequality_constraints=[],
            equality_constraints=[],
            acqf=self._acqf(),
            per_step_local_reopt=False,
        )
        assert valid is True
        assert variant[2].item() == pytest.approx(0.0)
        # other dims unchanged
        for j in (0, 1, 3):
            assert variant[j].item() == pytest.approx(0.5)

    def test_zero_variant_with_mixture_eq_projects_feasibly(self):
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        x = _row_to_tensor([0.4, 0.3, 0.2, 0.1])
        variant, valid = _build_variant(
            x,
            j_idx=2,
            kind=ActionKind.ZERO,
            bounds=inp["bounds"],
            inequality_constraints=inp["inequality_constraints"],
            equality_constraints=inp["equality_constraints"],
            acqf=self._acqf(),
            per_step_local_reopt=False,
        )
        assert valid is True
        assert variant[2].item() == pytest.approx(0.0, abs=1e-6)
        assert variant.sum().item() == pytest.approx(1.0, abs=1e-4)
        assert (variant >= 0).all()
        assert (variant <= 1).all()

    def test_zero_variant_with_inequality_only_respects_constraint(self):
        # x1 + x2 <= 0.6  →  rewritten by get_linear_constraints as -x1 - x2 >= -0.6
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                LinearInequalityConstraint(
                    features=["x1", "x2"],
                    coefficients=[1.0, 1.0],
                    rhs=0.6,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        x = _row_to_tensor([0.4, 0.4, 0.5, 0.5])  # already infeasible: x1+x2=0.8 > 0.6
        variant, valid = _build_variant(
            x,
            j_idx=0,
            kind=ActionKind.ZERO,
            bounds=inp["bounds"],
            inequality_constraints=inp["inequality_constraints"],
            equality_constraints=inp["equality_constraints"],
            acqf=self._acqf(),
            per_step_local_reopt=False,
        )
        assert valid is True
        assert variant[0].item() == pytest.approx(0.0, abs=1e-6)
        assert variant[0].item() + variant[1].item() <= 0.6 + 1e-4

    def test_zero_variant_qp_infeasible_returns_invalid_flag(self):
        # x1 + x2 = 1.0 with x2 bounds [0, 0.4]. Zeroing x1 forces
        # x2 = 1.0, which violates the upper bound — infeasible.
        inputs = [
            ContinuousInput(key="x1", bounds=(0.0, 1.0)),
            ContinuousInput(key="x2", bounds=(0.0, 0.4)),
        ]
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                LinearEqualityConstraint(
                    features=["x1", "x2"],
                    coefficients=[1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        x = _row_to_tensor([0.6, 0.4])
        variant, valid = _build_variant(
            x,
            j_idx=0,
            kind=ActionKind.ZERO,
            bounds=inp["bounds"],
            inequality_constraints=inp["inequality_constraints"],
            equality_constraints=inp["equality_constraints"],
            acqf=self._acqf(),
            per_step_local_reopt=False,
        )
        assert valid is False
        assert variant[0].item() == pytest.approx(0.0)

    def test_active_variant_with_mixture_eq_snaps_into_band(self):
        domain = _make_simple_domain(
            n_features=4,
            allow_zero=True,
            lb=0.2,
            ub=1.0,
            constraints=[
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        # x3 starts fractional at 0.05; snap to >= 0.2
        x = _row_to_tensor([0.5, 0.4, 0.05, 0.05])
        variant, valid = _build_variant(
            x,
            j_idx=2,
            kind=ActionKind.ACTIVE,
            bounds=inp["bounds"],
            inequality_constraints=inp["inequality_constraints"],
            equality_constraints=inp["equality_constraints"],
            acqf=self._acqf(),
            per_step_local_reopt=False,
            semicontinuous_specs=inp["semicontinuous_specs"],
        )
        assert valid is True
        assert variant[2].item() >= 0.2 - 1e-6
        assert variant.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_active_variant_no_constraints_keeps_starting_or_lb(self):
        # No linear constraints — projection short-circuits, we just clamp
        # the fractional column to lb.
        x = _row_to_tensor([0.5, 0.05, 0.5, 0.5])
        variant, valid = _build_variant(
            x,
            j_idx=1,
            kind=ActionKind.ACTIVE,
            bounds=torch.tensor([[0.0] * 4, [1.0] * 4], **tkwargs),
            inequality_constraints=[],
            equality_constraints=[],
            acqf=self._acqf(),
            per_step_local_reopt=False,
            semicontinuous_specs={1: (0.2, 1.0)},
        )
        assert valid is True
        assert variant[1].item() == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# End-to-end pruning tests
# ---------------------------------------------------------------------------


class TestPruneNchoosekEndToEnd:
    # ----- Single NChooseK without semi-continuity (matches today's path) -----

    def test_q1_max_count_violation_zeros_smallest_weight(self):
        domain, inp = _default_nchoosek_setup()
        # Weights: x2 and x3 have lowest, but only one needs to be zeroed
        weights = torch.tensor([10.0, 1.0, 2.0, 8.0], **tkwargs)
        acqf = WeightedSumAcqf(weights)
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, acqf),
            **inp,
            final_local_reopt=False,
        )
        # exactly two non-zero features
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz == 2
        # the zeroed features should be the lowest-weight ones (x2 then x3)
        assert out[0, 1].item() == pytest.approx(0.0)
        assert out[0, 2].item() == pytest.approx(0.0)

    def test_q1_already_feasible_is_noop(self):
        domain, inp = _default_nchoosek_setup()
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.5, 0.5, 0.0, 0.0]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        assert torch.allclose(out, X)

    def test_q2_processes_each_candidate(self):
        # Both candidates violate; both get pruned to <= 2 active.
        domain, inp = _default_nchoosek_setup()
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.4, 0.4, 0.4, 0.4],
            ]
        )
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        for i in (0, 1):
            nz = (out[i].abs() > 1e-6).sum().item()
            assert nz <= 2

    def test_q2_second_candidate_conditions_on_pruned_first(self):
        """During pruning of candidate i=1, every acquisition-function
        call must include the *pruned* X[0] in its prefix (not the
        dense X[0]). Locks in the q-batch joint-conditioning invariant
        the refactor preserves but the existing q2 test does not check.
        """
        domain, inp = _default_nchoosek_setup()
        rec = RecordingCallAF(MockAcquisitionFunction())
        X_dense = _stack_to_tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.4, 0.4, 0.4, 0.4],
            ]
        )
        out = prune_nchoosek(
            X=X_dense,
            acqf=cast(AcquisitionFunction, rec),
            **inp,
            final_local_reopt=False,
        )

        # Pruning must have changed X[0] for the test to be meaningful.
        assert not torch.allclose(out[0], X_dense[0])

        # Find AF calls that look like q-batch evaluations during the
        # pruning of i=1: shape (..., 2, d). Their first-position row
        # is the "prefix" — must equal the *pruned* X[0], never the
        # dense X[0].
        d = X_dense.shape[1]
        joint_calls = [c for c in rec.calls if c.dim() == 3 and c.shape[1] == 2]
        assert (
            joint_calls
        ), "expected at least one (..., 2, d) AF call during the inner loop for i=1"
        for c in joint_calls:
            assert c.shape[-1] == d
            prefix_row = c[..., 0, :]  # broadcast across the b-dim
            # Every batch-row's prefix row equals the pruned X[0].
            assert torch.allclose(
                prefix_row, out[0].expand_as(prefix_row)
            ), "prefix row must equal the pruned candidate-0, not any other value"
            # And it must NOT equal the dense X[0] (otherwise the
            # joint-conditioning is bypassed).
            assert not torch.allclose(prefix_row, X_dense[0].expand_as(prefix_row))

    # ----- Single NChooseK with linear overlap (today's QP path) -----

    def test_q1_with_mixture_constraint_qp_path(self):
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.4, 0.3, 0.2, 0.1]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz <= 2
        assert out[0].sum().item() == pytest.approx(1.0, abs=1e-3)

    # ----- Semi-continuous features -----

    def test_standalone_semicontinuous_no_nchoosek(self):
        # No NChooseK — only semi-continuity to enforce. The fractional
        # column must be resolved to either 0 or in [lb, ub].
        domain = _make_simple_domain(
            n_features=3,
            allow_zero=True,
            lb=0.2,
            ub=1.0,
        )
        inp = _inputs_from_domain(domain)
        # x2 starts at 0.1, in the gap (0, 0.2)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.5, 0.1, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        v = float(out[0, 1].abs().item())
        assert v <= 1e-6 or v >= 0.2 - 1e-6

    def test_semicontinuous_in_nchoosek_zero_variant_wins(self):
        # AF heavily favours x1 — fractional x4 should be zeroed (cheap).
        domain = _make_simple_domain(
            n_features=4,
            allow_zero=True,
            lb=0.2,
            ub=1.0,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        weights = torch.tensor([10.0, 5.0, 0.5, 0.1], **tkwargs)
        acqf = WeightedSumAcqf(weights)
        X = _stack_to_tensor([[0.5, 0.4, 0.05, 0.05]])
        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, acqf),
            **inp,
            final_local_reopt=False,
        )
        # x4 (lowest weight, fractional) should end at zero
        assert out[0, 3].item() == pytest.approx(0.0, abs=1e-3)
        # NChooseK satisfied
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz <= 2
        # mixture preserved
        assert out[0].sum().item() == pytest.approx(1.0, abs=1e-3)

    # ----- Multiple NChooseK -----

    def test_disjoint_nchoosek_constraints(self):
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                NChooseKConstraint(
                    features=["x3", "x4"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        active_x12 = (out[0, 0:2].abs() > 1e-6).sum().item()
        active_x34 = (out[0, 2:4].abs() > 1e-6).sum().item()
        assert active_x12 <= 1
        assert active_x34 <= 1

    def test_overlapping_nchoosek_one_zero_resolves_both(self):
        # Both constraints share x2. Zeroing x2 fixes both at once.
        # We force this by giving x1, x3 high weights and x2 low weight.
        domain = _make_simple_domain(
            n_features=3,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                NChooseKConstraint(
                    features=["x2", "x3"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        weights = torch.tensor([10.0, 0.1, 10.0], **tkwargs)
        acqf = WeightedSumAcqf(weights)
        X = _stack_to_tensor([[0.5, 0.5, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, acqf),
            **inp,
            final_local_reopt=False,
        )
        # x2 should be zeroed — the cheapest, fixes both
        assert out[0, 1].item() == pytest.approx(0.0)
        # both constraints satisfied
        assert (out[0, 0:2].abs() > 1e-6).sum().item() <= 1
        assert (out[0, 1:3].abs() > 1e-6).sum().item() <= 1

    def test_conflicting_nchoosek_raises_pruning_infeasible_error(self):
        # c1 demands min 2 active over {x1, x2}; c2 forbids both.
        domain = _make_simple_domain(
            n_features=2,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=2,
                    max_count=2,
                    none_also_valid=False,
                ),
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.5, 0.5]])
        with pytest.raises(PruningInfeasibleError):
            prune_nchoosek(
                X=X,
                acqf=acqf,
                **inp,
                final_local_reopt=False,
            )

    # ----- min_count guard -----

    def test_min_count_guard_blocks_zero_below_floor(self):
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=2,
                    max_count=4,
                    none_also_valid=False,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.5, 0.5, 0.0, 0.0]])  # at floor
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        # nothing should change — no max violation, no fractional
        assert torch.allclose(out, X)

    def test_min_count_guard_none_also_valid_allows_drop_to_zero(self):
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=2,
                    max_count=2,
                    none_also_valid=True,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        # Three actives — must zero one. Guard: post=2, OK.
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.0]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz == 2

    # ----- Hyperparameters -----

    def test_final_local_reopt_does_not_violate_constraints(self):
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.4, 0.3, 0.2, 0.1]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=True,
        )
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz <= 2
        assert out[0].sum().item() == pytest.approx(1.0, abs=1e-3)

    def test_per_step_local_reopt_runs_without_error(self):
        # Smoke-test: per_step_local_reopt=True doesn't break anything.
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.4, 0.3, 0.2, 0.1]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            per_step_local_reopt=True,
            final_local_reopt=False,
        )
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz <= 2
        assert out[0].sum().item() == pytest.approx(1.0, abs=1e-3)

    # ----- Failure paths and edge cases -----

    def test_q0_returns_unchanged(self):
        domain = _make_simple_domain(
            n_features=2,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = torch.empty((0, 2), **tkwargs)
        out = prune_nchoosek(X=X, acqf=acqf, **inp, final_local_reopt=False)
        assert out.shape == (0, 2)

    def test_constant_af_deterministic_selection(self):
        # ConstantAcqf returns the same value for every input → every
        # variant has identical absolute AF, every reduction is zero,
        # action selection falls back to the deterministic tie-break.
        domain, inp = _default_nchoosek_setup()
        acqf = ConstantAcqf(value=0.5)
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, acqf),
            **inp,
            final_local_reopt=False,
        )
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz == 2

    def test_qp_failure_does_not_crash(self, monkeypatch):
        # Force every QP call to raise; the pruning loop should still
        # return a tensor or raise PruningInfeasibleError, never crash.
        def fake_project(*args, **kwargs):
            raise RuntimeError("QP failed")

        monkeypatch.setattr(
            ncp,
            "project_to_feasible_space_via_slsqp",
            fake_project,
        )
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.4, 0.3, 0.2, 0.1]])
        # Either succeeds with -inf fallback or raises infeasibility —
        # but never an unhandled RuntimeError.
        try:
            prune_nchoosek(
                X=X,
                acqf=acqf,
                **inp,
                final_local_reopt=False,
            )
        except PruningInfeasibleError:
            pass

    def test_argmin_tie_break_returns_first_occurrence(self):
        # ConstantAcqf → all af_reductions are zero. Tie-break is
        # smallest j_idx with "zero" preferred over "active". With
        # 4 active features and max_count=2, we expect the first two
        # iterations to zero x1 and x2 (lowest j_idx).
        domain, inp = _default_nchoosek_setup()
        acqf = ConstantAcqf(value=1.0)
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, acqf),
            **inp,
            final_local_reopt=False,
        )
        # x1 and x2 should be zeroed (lowest indices, tie-break)
        assert out[0, 0].item() == pytest.approx(0.0)
        assert out[0, 1].item() == pytest.approx(0.0)
        # remaining are active
        assert out[0, 2].item() > 0
        assert out[0, 3].item() > 0

    # ----- Caller-supplied pinned_columns -----

    def test_fixed_feature_outside_nchoosek_does_not_move(self):
        # 5-feature domain: x5 is pinned at 0.7 by the caller. NChooseK
        # acts only on x1..x4. Mixture x1+...+x4 = 1 (does not include
        # the pinned feature). The pruning should not touch x5.
        inputs = [ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(4)]
        inputs.append(ContinuousInput(key="x5", bounds=(0.0, 1.0)))
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        # Candidate already carries the desired pinned value at x5.
        X = _stack_to_tensor([[0.4, 0.3, 0.2, 0.1, 0.7]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            pinned_columns={4},
            **inp,
            final_local_reopt=True,
        )
        assert out[0, 4].item() == pytest.approx(0.7, abs=1e-6)
        nz = (out[0, :4].abs() > 1e-6).sum().item()
        assert nz <= 2
        assert out[0, :4].sum().item() == pytest.approx(1.0, abs=1e-3)

    def test_fixed_zero_feature_does_not_count_or_move(self):
        # x5 is pinned to zero by the caller — it should never be a
        # zero-action target (it is already zero by external decree)
        # and the final value must remain exactly 0.
        inputs = [ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(5)]
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        # Candidate already carries x5 = 0 (the desired pinned value).
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5, 0.0]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            pinned_columns={4},
            **inp,
            final_local_reopt=True,
        )
        assert out[0, 4].item() == pytest.approx(0.0)
        nz = (out[0, :4].abs() > 1e-6).sum().item()
        assert nz <= 2

    def test_fixed_feature_inside_nchoosek_excluded_from_actions(self):
        # x1 is pinned at 0.5 even though it appears in the NChooseK
        # constraint. The pruning must exclude it from the action set
        # (cannot zero a pinned feature) and instead zero among
        # {x2, x3, x4}.
        domain, inp = _default_nchoosek_setup()
        # Use weights that would otherwise make x1 the prime zero
        # candidate (smallest weight). The pinning filter must
        # override.
        weights = torch.tensor([0.1, 5.0, 5.0, 10.0], **tkwargs)
        acqf = WeightedSumAcqf(weights)
        # Candidate already carries x1 = 0.5 (the desired pinned value).
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, acqf),
            pinned_columns={0},
            **inp,
            final_local_reopt=False,
        )
        assert out[0, 0].item() == pytest.approx(0.5)
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz <= 2


# ---------------------------------------------------------------------------
# Partial-drift fix tests
# ---------------------------------------------------------------------------


class TestPartialDriftFix:
    """Verify that variant builders tighten bounds for previously-committed
    active semi-continuous features so they cannot drift into the
    semi-continuity gap during a later iteration's optimize_acqf or QP
    projection. Covers both `_build_zero_variant` and
    `_build_active_variant`, plus the deactivation case (the `−{j_idx}`
    exclusion) and an end-to-end `prune_nchoosek` run with
    `final_local_reopt=False`.
    """

    def _setup_mixture_4(self):
        """4 semi-continuous features (lb=0.2, ub=1) in a mixture
        Σ x = 1. Returns (bounds, ineq, eq, semi_specs) ready for use
        with the variant builders.
        """
        d = 4
        bounds = torch.tensor(
            [[0.0] * d, [1.0] * d],
            **tkwargs,
        )
        # Σ x = 1, expressed in get_linear_constraints' format
        # (indices, -coefficients, -rhs):
        eq = [
            (
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([-1.0, -1.0, -1.0, -1.0], **tkwargs),
                -1.0,
            )
        ]
        ineq: list = []
        semi = {i: (0.2, 1.0) for i in range(d)}
        return bounds, ineq, eq, semi

    def test_zero_variant_keeps_active_semi_in_band(self):
        """Building a zero variant for a feature while two semi-continuous
        features are committed-active: those committed actives must stay
        in `[0.2, 1]` after the projection, not drift into `(0, 0.2)`.
        """
        bounds, ineq, eq, semi = self._setup_mixture_4()
        # x_1 and x_2 are already in their active bands [0.2, 1]; x_0
        # is the deactivation target; x_3 is zero. The candidate is
        # mixture-feasible (sums to 1).
        x_i = torch.tensor([0.2, 0.4, 0.4, 0.0], **tkwargs)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        variant, valid = _build_variant(
            x_i,
            j_idx=0,
            kind=ActionKind.ZERO,
            bounds=bounds,
            inequality_constraints=ineq,
            equality_constraints=eq,
            acqf=acqf,
            per_step_local_reopt=False,
            active_set={1, 2},
            semicontinuous_specs=semi,
        )
        assert valid is True
        # x_0 was the target → 0
        assert variant[0].item() == pytest.approx(0.0, abs=1e-6)
        # x_1 and x_2 were committed-active → must stay in [0.2, 1]
        assert variant[1].item() >= 0.2 - 1e-6
        assert variant[1].item() <= 1.0 + 1e-6
        assert variant[2].item() >= 0.2 - 1e-6
        assert variant[2].item() <= 1.0 + 1e-6
        # mixture preserved
        assert variant.sum().item() == pytest.approx(1.0, abs=1e-3)

    def test_active_variant_keeps_other_active_semi_in_band(self):
        """Building an active variant for a fractional feature while
        another semi-continuous feature is already committed-active:
        the committed-active one must keep its band.
        """
        bounds, ineq, eq, semi = self._setup_mixture_4()
        # x_1 already active at 0.4; x_2 is fractional (0.1) and we'll
        # commit it active. x_0 free, x_3 zero.
        x_i = torch.tensor([0.5, 0.4, 0.1, 0.0], **tkwargs)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        variant, valid = _build_variant(
            x_i,
            j_idx=2,
            kind=ActionKind.ACTIVE,
            bounds=bounds,
            inequality_constraints=ineq,
            equality_constraints=eq,
            acqf=acqf,
            per_step_local_reopt=False,
            active_set={1},
            semicontinuous_specs=semi,
        )
        assert valid is True
        # x_2 (target) snapped into [0.2, 1]
        assert variant[2].item() >= 0.2 - 1e-6
        # x_1 (committed-active) stays in [0.2, 1]
        assert variant[1].item() >= 0.2 - 1e-6
        assert variant[1].item() <= 1.0 + 1e-6
        # mixture preserved
        assert variant.sum().item() == pytest.approx(1.0, abs=1e-3)

    def test_zero_variant_can_deactivate_active_semi(self):
        """Building a zero variant for a feature that is itself
        currently in `active_set` and is semi-continuous: the `− {j_idx}`
        exclusion must let it leave its band and pin to 0, while *other*
        active semi features stay locked.
        """
        bounds, ineq, eq, semi = self._setup_mixture_4()
        # Three actives all at 0.33; x_3 zero. We deactivate x_2.
        x_i = torch.tensor(
            [0.33, 0.34, 0.33, 0.0],
            **tkwargs,
        )
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        variant, valid = _build_variant(
            x_i,
            j_idx=2,
            kind=ActionKind.ZERO,
            bounds=bounds,
            inequality_constraints=ineq,
            equality_constraints=eq,
            acqf=acqf,
            per_step_local_reopt=False,
            active_set={0, 1, 2},
            semicontinuous_specs=semi,
        )
        assert valid is True
        # x_2 (the deactivation target) pinned to 0 — exclusion did its
        # job; we did NOT keep x_2 ≥ 0.2.
        assert variant[2].item() == pytest.approx(0.0, abs=1e-6)
        # x_0 and x_1 (still active) stay in [0.2, 1]
        assert variant[0].item() >= 0.2 - 1e-6
        assert variant[1].item() >= 0.2 - 1e-6
        # mixture preserved
        assert variant.sum().item() == pytest.approx(1.0, abs=1e-3)

    def test_pruning_no_drift_with_flr_false(self):
        """End-to-end: a 4-feature semi-continuous mixture domain.
        With `final_local_reopt=False`, every coordinate of the final
        candidate must lie in `{0} ∪ [lb, ub]`. Without the partial-
        drift fix the per-step reopts could leave coordinates in
        `(0, lb)` and `flr=False` wouldn't catch them.
        """
        domain = _make_simple_domain(
            n_features=4,
            allow_zero=True,
            lb=0.2,
            ub=1.0,
            constraints=[
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=0,
                    max_count=4,
                    none_also_valid=True,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        # Initial candidate: all four equal at 0.25 (every coordinate
        # is fractional, in (0, 0.2)? Actually 0.25 > 0.2 so they're
        # cleanly active. Use a fractional starting point instead so
        # the active variant is exercised.)
        X = _stack_to_tensor([[0.1, 0.4, 0.4, 0.1]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        for j in range(4):
            v = float(out[0, j].abs().item())
            assert v <= 1e-6 or v >= 0.2 - 1e-6, (
                f"x_{j + 1}={v} fell into the (0, 0.2) gap; "
                "partial-drift fix didn't keep it in band."
            )


# ---------------------------------------------------------------------------
# Activate-zero tests (min_count > 0)
# ---------------------------------------------------------------------------


class TestActivateZero:
    """Verify the activate-zero action: when an NChooseK constraint has
    ``a_c < min_count_c``, the loop activates currently-zero features
    rather than raising. Covers the new helpers, the variant builder,
    the loop's eligibility logic, and end-to-end formulation cases.
    """

    def _basic_constraint(self, **kw) -> NChooseKConstraint:
        defaults = {
            "features": ["x1", "x2", "x3", "x4"],
            "min_count": 2,
            "max_count": 3,
            "none_also_valid": False,
        }
        defaults.update(kw)
        return NChooseKConstraint(**defaults)  # type: ignore

    def _f2i_continuous(self, n: int) -> Dict[str, Tuple[int, ...]]:
        return {f"x{i + 1}": (i,) for i in range(n)}

    # ----- _min_count_violated_constraints -----

    def test_min_count_violated_marks_count_below_min(self):
        c = self._basic_constraint(min_count=3, max_count=4)
        # count=2 < min=3, none_also_valid=False
        violated = _min_count_violated_constraints({0: 2}, [c])
        assert violated == {0}

    def test_min_count_violated_satisfied_at_min(self):
        c = self._basic_constraint(min_count=3, max_count=4)
        violated = _min_count_violated_constraints({0: 3}, [c])
        assert violated == set()

    def test_min_count_none_also_valid_at_zero(self):
        c = self._basic_constraint(min_count=3, max_count=4, none_also_valid=True)
        violated = _min_count_violated_constraints({0: 0}, [c])
        assert violated == set()

    def test_min_count_none_also_valid_above_zero_below_min(self):
        c = self._basic_constraint(min_count=3, max_count=4, none_also_valid=True)
        # count=1 between 0 and min: still violated, the carve-out only
        # exempts exactly count=0.
        violated = _min_count_violated_constraints({0: 1}, [c])
        assert violated == {0}

    # ----- _features_eligible_for_activate -----

    def test_features_eligible_for_activate_only_zero_features_in_violated(
        self,
    ):
        c1 = self._basic_constraint(features=["x1", "x2"], min_count=2, max_count=2)
        c2 = self._basic_constraint(features=["x3", "x4"], min_count=0, max_count=2)
        f2i = self._f2i_continuous(4)
        # zero_set = {0, 1, 2}, only c1 (idx=0) is violated;
        # c1 features are x1, x2 → indices {0, 1}; intersect with
        # zero_set = {0, 1}.
        eligible = _features_eligible_for_activate(
            zero_set={0, 1, 2},
            min_count_violated={0},
            nchoosek_constraints=[c1, c2],
            features2idx=f2i,
        )
        assert eligible == {0, 1}

    def test_features_eligible_for_activate_empty_when_no_violation(self):
        c = self._basic_constraint()
        f2i = self._f2i_continuous(4)
        assert (
            _features_eligible_for_activate(
                zero_set={0, 1, 2},
                min_count_violated=set(),
                nchoosek_constraints=[c],
                features2idx=f2i,
            )
            == set()
        )

    # ----- _activate_action_blocked_by_max_count -----

    def test_activate_blocked_when_at_max_count(self):
        # max_count=2 already, activating one more pushes to 3 → blocked.
        c = self._basic_constraint(min_count=0, max_count=2)
        f2i = self._f2i_continuous(4)
        assert (
            _activate_action_blocked_by_max_count(
                j_idx=0,
                active_counts={0: 2},
                nchoosek_constraints=[c],
                features2idx=f2i,
            )
            is True
        )

    def test_activate_not_blocked_below_max(self):
        c = self._basic_constraint(min_count=0, max_count=3)
        f2i = self._f2i_continuous(4)
        assert (
            _activate_action_blocked_by_max_count(
                j_idx=0,
                active_counts={0: 1},
                nchoosek_constraints=[c],
                features2idx=f2i,
            )
            is False
        )

    def test_activate_not_blocked_when_feature_outside_constraint(self):
        # j_idx=3 (x4) is not in c1 → guard skips it.
        c1 = NChooseKConstraint(  # type: ignore
            features=["x1", "x2"],
            min_count=0,
            max_count=2,
            none_also_valid=False,
        )
        f2i = self._f2i_continuous(4)
        assert (
            _activate_action_blocked_by_max_count(
                j_idx=3,
                active_counts={0: 2},  # c1 at max
                nchoosek_constraints=[c1],
                features2idx=f2i,
            )
            is False
        )

    # ----- _build_activate_variant -----

    def test_build_activate_variant_non_semicontinuous(self):
        # 4-feature mixture, all non-semicontinuous.
        bounds = torch.tensor([[0.0] * 4, [1.0] * 4], **tkwargs)
        eq = [
            (
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([-1.0] * 4, **tkwargs),
                -1.0,
            )
        ]
        # Two features active (x_1, x_2 at 0.5 each); x_3, x_4 zero.
        # Activate x_3.
        x_i = torch.tensor([0.0, 0.5, 0.5, 0.0], **tkwargs)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        variant, valid = _build_variant(
            x_i,
            j_idx=2,
            kind=ActionKind.ACTIVATE,
            bounds=bounds,
            inequality_constraints=[],
            equality_constraints=eq,
            acqf=acqf,
            per_step_local_reopt=False,
            pinned_zero_indices={0, 3},  # zero_set − {j_idx}
            active_set={1},
            semicontinuous_specs={},
            tol=1e-6,
        )
        assert valid is True
        # x_2 (target) ≥ tol (positive)
        assert variant[2].item() >= 1e-6
        # mixture preserved
        assert variant.sum().item() == pytest.approx(1.0, abs=1e-3)
        # other zeros stayed zero
        assert variant[0].item() == pytest.approx(0.0, abs=1e-6)
        assert variant[3].item() == pytest.approx(0.0, abs=1e-6)

    def test_build_activate_variant_semicontinuous(self):
        # 4 semi-continuous features (lb=0.2, ub=1) + mixture.
        bounds = torch.tensor([[0.0] * 4, [1.0] * 4], **tkwargs)
        eq = [
            (
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([-1.0] * 4, **tkwargs),
                -1.0,
            )
        ]
        semi = {i: (0.2, 1.0) for i in range(4)}
        # x_1, x_2 active (in band); x_3, x_4 zero.
        x_i = torch.tensor([0.0, 0.4, 0.6, 0.0], **tkwargs)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        variant, valid = _build_variant(
            x_i,
            j_idx=2,
            kind=ActionKind.ACTIVATE,
            bounds=bounds,
            inequality_constraints=[],
            equality_constraints=eq,
            acqf=acqf,
            per_step_local_reopt=False,
            pinned_zero_indices={0, 3},
            active_set={1},
            semicontinuous_specs=semi,
            tol=1e-6,
        )
        assert valid is True
        # x_2 must enter its semi-continuous band [0.2, 1].
        assert variant[2].item() >= 0.2 - 1e-6
        # x_1 (committed-active) stays in band.
        assert variant[1].item() >= 0.2 - 1e-6
        # mixture preserved
        assert variant.sum().item() == pytest.approx(1.0, abs=1e-3)

    # ----- End-to-end -----

    def test_pruning_min_count_formulation(self):
        """End-to-end: 4-feature mixture domain with min_count=3,
        max_count=4. AF places mass on 2 features. Pruning should
        activate to satisfy min_count.
        """
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=3,
                    max_count=4,
                    none_also_valid=False,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        # AF maximiser placed mass on only 2 features.
        X = _stack_to_tensor([[0.4, 0.6, 0.0, 0.0]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        # NChooseK satisfied: 3 or 4 active.
        nz = (out[0].abs() > 1e-6).sum().item()
        assert 3 <= nz <= 4
        # mixture preserved
        assert out[0].sum().item() == pytest.approx(1.0, abs=1e-3)

    def test_pruning_min_count_with_none_also_valid_does_not_zero_down(self):
        """When the constraint has none_also_valid=True and the
        candidate is at 0 < count < min, we activate up rather than
        attempting to zero down. (The per-step zero-guard blocks the
        zero-down trajectory.)
        """
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=3,
                    max_count=4,
                    none_also_valid=True,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.4, 0.6, 0.0, 0.0]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            **inp,
            final_local_reopt=False,
        )
        # The algorithm activated up to ≥3 active rather than zeroing
        # down to 0. Either count=3 or count=4 is acceptable; count=0
        # would also satisfy the constraint but is unreachable from
        # count>0 by the per-step rules.
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz >= 3, f"expected count ≥ 3 (activation), got {nz}"
        assert nz <= 4
        assert out[0].sum().item() == pytest.approx(1.0, abs=1e-3)

    def test_pruning_conflicting_min_max_raises(self):
        """Two NChooseK constraints over the same features with mutually
        infeasible min/max: min_count=3 forces activation, max_count=1
        forbids it. PruningInfeasibleError should fire.
        """
        domain = _make_simple_domain(
            n_features=4,
            constraints=[
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=3,
                    max_count=4,
                    none_also_valid=False,
                ),
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=False,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[1.0, 0.0, 0.0, 0.0]])
        with pytest.raises(PruningInfeasibleError):
            prune_nchoosek(
                X=X,
                acqf=acqf,
                **inp,
                final_local_reopt=False,
            )


class TestDataclasses:
    def test_action_kind_ordering_matches_legacy_priority(self):
        # Tie-break: ZERO beats ACTIVE beats ACTIVATE.
        assert ActionKind.ZERO.value < ActionKind.ACTIVE.value
        assert ActionKind.ACTIVE.value < ActionKind.ACTIVATE.value

    def test_action_holds_supplied_fields(self):
        v = torch.tensor([1.0, 0.0])
        a = Action(j=2, kind=ActionKind.ZERO, variant=v, valid=True)
        assert a.j == 2
        assert a.kind is ActionKind.ZERO
        assert torch.equal(a.variant, v)
        assert a.valid is True

    def test_pruning_state_commit_zero_moves_feature_to_zero_set(self):
        s = PruningState(
            x=torch.zeros(4),
            zero_set=set(),
            frac_set={1},
            active_set={0, 2},
        )
        s.commit(Action(j=2, kind=ActionKind.ZERO, variant=torch.zeros(4), valid=True))
        assert s.zero_set == {2}
        assert s.frac_set == {1}
        assert s.active_set == {0}

    def test_pruning_state_commit_active_moves_frac_to_active(self):
        s = PruningState(
            x=torch.zeros(4),
            zero_set=set(),
            frac_set={1},
            active_set={0},
        )
        s.commit(
            Action(j=1, kind=ActionKind.ACTIVE, variant=torch.zeros(4), valid=True)
        )
        assert s.frac_set == set()
        assert s.active_set == {0, 1}

    def test_pruning_state_commit_activate_moves_zero_to_active(self):
        s = PruningState(
            x=torch.zeros(4),
            zero_set={2, 3},
            frac_set=set(),
            active_set={0, 1},
        )
        s.commit(
            Action(
                j=2,
                kind=ActionKind.ACTIVATE,
                variant=torch.zeros(4),
                valid=True,
            )
        )
        assert s.zero_set == {3}
        assert s.active_set == {0, 1, 2}


class TestXPendingThreading:
    """Regression tests for the X_pending threading into the local-reopt
    path. The threading must:

    1. Preserve any pre-existing ``acqf.X_pending`` (set at AF
       construction time, e.g. via ``strategy.set_candidates(...)``).
    2. Concatenate the pruning prefix onto it during
       ``_local_optacqf`` calls.
    3. Restore the original ``X_pending`` afterwards via try/finally.
    4. Silently no-op for analytic AFs that raise ``UnsupportedError``
       on ``set_X_pending`` (their joint AF == marginal on a q-batch).
    """

    @staticmethod
    def _two_candidate_setup() -> Tuple[Any, Any, Tensor]:
        domain, inp = _default_nchoosek_setup()
        X = _stack_to_tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.4, 0.4, 0.4, 0.4],
            ]
        )
        return domain, inp, X

    def test_x_pending_restored_after_prune(self):
        """``acqf.X_pending`` must equal its pre-call value after
        ``prune_nchoosek`` returns, regardless of how many local-reopt
        calls happen inside.
        """
        _, inp, X = self._two_candidate_setup()
        rec, _ = _make_recording_pending_af(MockAcquisitionFunction())
        # Pre-set a known X_pending on the AF (mimics
        # strategy.set_candidates → AF construction).
        user_pending = torch.tensor([[0.1, 0.2, 0.3, 0.4]], **tkwargs)
        rec.X_pending = user_pending.clone()

        prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, rec),
            **inp,
            per_step_local_reopt=True,
            final_local_reopt=True,
        )

        # After the call, X_pending must be the pre-set tensor again.
        assert rec.X_pending is not None
        assert torch.allclose(rec.X_pending, user_pending)

    def test_x_pending_concat_during_local_optacqf(self):
        """During the per-step reopt of i=1, ``set_X_pending`` must be
        called with ``cat([user_pending, X_prefix_i1])``, not just
        ``X_prefix_i1`` (which would silently overwrite the user's
        pending candidates).
        """
        _, inp, X = self._two_candidate_setup()
        rec, calls = _make_recording_pending_af(MockAcquisitionFunction())
        user_pending = torch.tensor([[0.1, 0.2, 0.3, 0.4]], **tkwargs)
        rec.X_pending = user_pending.clone()

        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, rec),
            **inp,
            per_step_local_reopt=True,
            final_local_reopt=True,
        )

        # X[0] must have been pruned (otherwise the test is vacuous).
        assert not torch.allclose(out[0], X[0])

        # Must have at least one set_X_pending call where the value
        # has shape (1 + |X_prefix_i1|, d) = (2, d) — i.e., user_pending
        # concatenated with the pruned X[0].
        d = X.shape[1]
        concat_calls = [
            v
            for tag, v in calls
            if tag == "set" and v is not None and v.shape == (2, d)
        ]
        assert concat_calls, (
            "expected at least one set_X_pending call with shape "
            "(2, d) — concat of user_pending and pruned X[0]"
        )
        # Every such call must carry user_pending in the first row
        # and the pruned X[0] in the second row.
        for v in concat_calls:
            assert torch.allclose(v[0], user_pending[0])
            assert torch.allclose(v[1], out[0])

    def test_x_pending_no_user_pending(self):
        """When ``acqf.X_pending`` starts as None, the local-reopt
        path must call ``set_X_pending`` with just the prefix (not
        ``cat([None, prefix])``, which would crash).
        """
        _, inp, X = self._two_candidate_setup()
        rec, calls = _make_recording_pending_af(MockAcquisitionFunction())
        rec.X_pending = None  # no user-set pending

        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, rec),
            **inp,
            per_step_local_reopt=True,
            final_local_reopt=True,
        )

        d = X.shape[1]
        # Calls during i=1 must be shape (1, d) — the pruned X[0]
        # alone, no user_pending to concat.
        prefix_only_calls = [
            v
            for tag, v in calls
            if tag == "set" and v is not None and v.shape == (1, d)
        ]
        assert prefix_only_calls, (
            "expected at least one set_X_pending call with shape "
            "(1, d) — the pruned X[0] as the entire pending set"
        )
        for v in prefix_only_calls:
            assert torch.allclose(v[0], out[0])

        # And after the run X_pending is restored to None.
        assert rec.X_pending is None

    def test_final_local_reopt_threads_pending(self):
        """The final polish step must also receive the prefix via
        ``X_pending_extra`` — verified by recording set_X_pending
        calls when only ``final_local_reopt=True`` (per-step disabled).
        """
        _, inp, X = self._two_candidate_setup()
        rec, calls = _make_recording_pending_af(MockAcquisitionFunction())
        user_pending = torch.tensor([[0.1, 0.2, 0.3, 0.4]], **tkwargs)
        rec.X_pending = user_pending.clone()

        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, rec),
            **inp,
            per_step_local_reopt=False,
            final_local_reopt=True,
        )

        d = X.shape[1]
        # With per_step_local_reopt=False, the only local-reopt path
        # exercised is _final_local_reopt for i=1 (i=0 has empty
        # prefix so no set_X_pending call is made there). It must
        # have called set_X_pending with concat([user_pending,
        # pruned_X[0]]) — shape (2, d).
        concat_calls = [
            v
            for tag, v in calls
            if tag == "set" and v is not None and v.shape == (2, d)
        ]
        assert concat_calls, (
            "expected at least one set_X_pending call with shape "
            "(2, d) during the final polish for i=1"
        )
        for v in concat_calls:
            assert torch.allclose(v[0], user_pending[0])
            assert torch.allclose(v[1], out[0])


class TestPinnedColumns:
    """Regression tests for the ``pinned_columns`` API: tensor columns
    that must remain at the candidate's per-row value through every
    QP projection and reopt call. The mechanism replaces the old
    ``fixed_features: Dict[int, float]`` parameter and supports
    per-row resolution (different candidates in a q-batch can carry
    different values for the same pinned column).
    """

    def test_pinned_column_not_proposed_as_action(self):
        """A pinned column inside an NChooseK feature set must be
        excluded from the action set. Otherwise the loop would try
        to zero it (smallest AF impact via the weights) and the
        pinning would silently fail.
        """
        domain, inp = _default_nchoosek_setup()
        # Weights make x1 the AF-cheapest zero target.
        weights = torch.tensor([0.1, 5.0, 5.0, 10.0], **tkwargs)
        acqf = WeightedSumAcqf(weights)
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, acqf),
            pinned_columns={0},
            **inp,
            final_local_reopt=False,
        )
        # x1 (column 0) must still be 0.5; pruning must have zeroed
        # one of {x2, x3, x4} instead.
        assert out[0, 0].item() == pytest.approx(0.5)

    def test_pinned_column_value_unchanged_q1(self):
        """q=1, mixture sums over four NChooseK features, plus a
        fifth column pinned at 0.7 outside the NChooseK. The value
        on the pinned column survives the QP projection exactly.
        """
        inputs = [ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(5)]
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.4, 0.3, 0.2, 0.1, 0.7]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            pinned_columns={4},
            **inp,
            final_local_reopt=False,
        )
        assert out[0, 4].item() == pytest.approx(0.7, abs=1e-9)

    def test_pinned_column_value_unchanged_per_row_q2(self):
        """q=2 with the same column pinned to *different* values
        across rows. Locks in the per-row resolution: row 0 carries
        0.3, row 1 carries 0.8, both must survive pruning unchanged.
        """
        inputs = [ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(5)]
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        # Same column 4, different per-row values.
        X = _stack_to_tensor(
            [
                [0.5, 0.5, 0.5, 0.5, 0.3],
                [0.4, 0.4, 0.4, 0.4, 0.8],
            ]
        )
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            pinned_columns={4},
            **inp,
            final_local_reopt=False,
        )
        assert out[0, 4].item() == pytest.approx(0.3, abs=1e-9)
        assert out[1, 4].item() == pytest.approx(0.8, abs=1e-9)

    def test_pinned_column_propagates_to_final_local_reopt(self):
        """``final_local_reopt=True`` runs an additional
        ``optimize_acqf`` polish; the pinned column must still survive.
        """
        inputs = [ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(5)]
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5, 0.7]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            pinned_columns={4},
            **inp,
            final_local_reopt=True,
        )
        assert out[0, 4].item() == pytest.approx(0.7, abs=1e-6)

    def test_pinned_column_propagates_to_per_step_local_reopt(self):
        """``per_step_local_reopt=True`` runs ``optimize_acqf`` per
        action variant; pin must hold through every reopt call.
        """
        inputs = [ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(5)]
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
            ],
        )
        inp = _inputs_from_domain(domain)
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5, 0.7]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            pinned_columns={4},
            **inp,
            per_step_local_reopt=True,
            final_local_reopt=False,
        )
        assert out[0, 4].item() == pytest.approx(0.7, abs=1e-6)

    def test_categorical_one_hot_survives_pruning(self):
        """Integration test: a domain with a one-hot-encoded
        ``CategoricalInput`` plus an NChooseK over four continuous
        features. Build ``pinned_columns`` the way
        ``BotorchOptimizer._prune`` would (every column
        outside un-fixed ``ContinuousInput``s) and run
        ``prune_nchoosek`` end-to-end with ``final_local_reopt=True``.
        Assert the categorical's three one-hot columns survive the
        full pruning pipeline (QP projection + final polish) without
        drifting to fractional values.
        """
        inputs: List[Any] = [
            ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(4)
        ]
        inputs.append(CategoricalInput(key="cat", categories=["A", "B", "C"]))
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )

        # Force one-hot encoding so the categorical occupies multiple
        # columns — the nasty drift case.
        specs: Dict[str, Any] = {"cat": OneHotEncoding()}
        features2idx, _ = domain.inputs._get_transform_info(specs)
        bounds = get_torch_bounds_from_domain(domain, specs)
        inequality_constraints = get_linear_constraints(
            domain, constraint=LinearInequalityConstraint
        )
        equality_constraints = get_linear_constraints(
            domain, constraint=LinearEqualityConstraint
        )
        nchoosek_constraints = list(domain.constraints.get(NChooseKConstraint))

        # Mirror BotorchOptimizer._prune.
        pinned_columns: set = set()
        for feat in domain.inputs:
            cols = features2idx[feat.key]
            if isinstance(feat, ContinuousInput) and feat.fixed_value() is None:
                continue
            pinned_columns.update(cols)

        cat_cols = list(features2idx["cat"])
        assert set(cat_cols) <= pinned_columns

        # Candidate: continuous mixture x1+...+x4 = 1, categorical
        # one-hot = (0, 1, 0) for category B.
        d = bounds.shape[1]
        row = torch.zeros(d, **tkwargs)
        row[0:4] = torch.tensor([0.4, 0.3, 0.2, 0.1], **tkwargs)
        # cat_cols[1] is category B's column — set its one-hot to 1.
        row[cat_cols[1]] = 1.0

        X = row.unsqueeze(0).clone()
        X_in = X.clone()  # snapshot for comparison

        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())

        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            nchoosek_constraints=nchoosek_constraints,
            features2idx=features2idx,
            bounds=bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            semicontinuous_specs={},
            pinned_columns=pinned_columns,
            final_local_reopt=True,
        )

        # Categorical one-hot columns must be exactly preserved —
        # no drift to fractional values from QP projection or polish.
        for col in cat_cols:
            assert torch.isclose(out[0, col], X_in[0, col], atol=1e-9), (
                f"categorical column {col} drifted from "
                f"{X_in[0, col].item()} to {out[0, col].item()}"
            )
        # Sanity: continuous mixture still satisfied; NChooseK pruning happened.
        assert out[0, :4].sum().item() == pytest.approx(1.0, abs=1e-3)
        nz = (out[0, :4].abs() > 1e-6).sum().item()
        assert nz <= 2

    def test_interpoint_feature_pinned_through_pruning(self):
        """A domain with NChooseK over continuous features plus an
        ``InterpointEqualityConstraint`` over a disjoint continuous
        feature. The interpoint feature must not drift during pruning
        — the QP projection only sees linear constraints, so without
        the caller pinning the interpoint feature column, SLSQP would
        be free to move it within its bounds and silently break the
        interpoint constraint.

        Mirrors ``BotorchOptimizer._prune``'s pinning
        construction, including the new loop that covers
        ``Interpoint`` / ``Nonlinear`` / ``Product`` constraint
        features.
        """
        inputs: List[Any] = [
            ContinuousInput(key=f"x{i + 1}", bounds=(0.0, 1.0)) for i in range(4)
        ]
        inputs.append(ContinuousInput(key="x5", bounds=(0.0, 1.0)))
        outputs = [ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))]
        domain = Domain.from_lists(
            inputs=inputs,
            outputs=outputs,
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    min_count=1,
                    max_count=2,
                    none_also_valid=False,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2", "x3", "x4"],
                    coefficients=[1.0, 1.0, 1.0, 1.0],
                    rhs=1.0,
                ),
                InterpointEqualityConstraint(features=["x5"]),
            ],
        )

        specs: Dict[str, Any] = {}
        features2idx, _ = domain.inputs._get_transform_info(specs)
        bounds = get_torch_bounds_from_domain(domain, specs)
        inequality_constraints = get_linear_constraints(
            domain, constraint=LinearInequalityConstraint
        )
        equality_constraints = get_linear_constraints(
            domain, constraint=LinearEqualityConstraint
        )
        nchoosek_constraints = list(domain.constraints.get(NChooseKConstraint))

        # Mirror BotorchOptimizer._prune's full pinning
        # logic, including the new Interpoint/Nonlinear/Product loop.
        pinned_columns: set = set()
        for feat in domain.inputs:
            cols = features2idx[feat.key]
            if isinstance(feat, ContinuousInput) and feat.fixed_value() is None:
                continue
            pinned_columns.update(cols)
        for c in domain.constraints.get(
            includes=[InterpointConstraint, NonlinearConstraint, ProductConstraint]
        ):
            for feat_key in c.features:
                pinned_columns.update(features2idx[feat_key])

        # The interpoint feature column must be in pinned_columns.
        x5_col = features2idx["x5"][0]
        assert x5_col in pinned_columns

        # Candidate carries x5 = 0.7 (the value the interpoint
        # constraint would lock across the q-batch).
        X = _stack_to_tensor([[0.4, 0.3, 0.2, 0.1, 0.7]])
        acqf = cast(AcquisitionFunction, MockAcquisitionFunction())

        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            nchoosek_constraints=nchoosek_constraints,
            features2idx=features2idx,
            bounds=bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            semicontinuous_specs={},
            pinned_columns=pinned_columns,
            final_local_reopt=True,
        )

        # x5 unchanged through both QP projection and final polish.
        assert out[0, x5_col].item() == pytest.approx(0.7, abs=1e-9)
        # NChooseK pruning still worked on x1..x4.
        nz = (out[0, :4].abs() > 1e-6).sum().item()
        assert nz <= 2
        assert out[0, :4].sum().item() == pytest.approx(1.0, abs=1e-3)


class TestIsNchoosekPruningApplicable:
    def test_no_nchoosek(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is False

    def test_nchoosek_only(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
                ContinuousInput(key="x3", bounds=(0, 1)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3"],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is True

    def test_nchoosek_with_overlapping_linear_inequality(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
                ContinuousInput(key="x3", bounds=(0, 1)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3"],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
                LinearInequalityConstraint(
                    features=["x1", "x2"],
                    coefficients=[1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is True
        assert has_nchoosek_linear_overlap(domain) is True

    def test_nchoosek_with_overlapping_linear_equality(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
                ContinuousInput(key="x3", bounds=(0, 1)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3"],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
                LinearEqualityConstraint(
                    features=["x1", "x2"],
                    coefficients=[1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is True
        assert has_nchoosek_linear_overlap(domain) is True

    def test_nchoosek_with_partial_linear_overlap(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
                ContinuousInput(key="x3", bounds=(0, 1)),
                ContinuousInput(key="x4", bounds=(0, 1)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3"],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
                LinearInequalityConstraint(
                    features=["x3", "x4"],
                    coefficients=[1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is True
        assert has_nchoosek_linear_overlap(domain) is True

    def test_nchoosek_with_truly_disjoint_linear(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
                ContinuousInput(key="x3", bounds=(0, 1)),
                ContinuousInput(key="x4", bounds=(0, 1)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                LinearInequalityConstraint(
                    features=["x3", "x4"],
                    coefficients=[1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is True
        assert has_nchoosek_linear_overlap(domain) is False

    def test_nchoosek_blocked_by_product_constraint(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
                ContinuousInput(key="x3", bounds=(0, 1)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3"],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
                ProductInequalityConstraint(
                    features=["x1", "x2"],
                    exponents=[1.0, 1.0],
                    rhs=0.5,
                    sign=1,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is False

    def test_multiple_nchoosek_one_overlapping_linear(self):
        domain = Domain.from_lists(
            inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(1, 5)],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                NChooseKConstraint(
                    features=["x3", "x4"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                LinearInequalityConstraint(
                    features=["x3", "x4"],
                    coefficients=[1.0, 1.0],
                    rhs=1.0,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is True

    def test_multiple_nchoosek_blocked_by_product(self):
        domain = Domain.from_lists(
            inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(1, 5)],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                NChooseKConstraint(
                    features=["x3", "x4"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                ProductInequalityConstraint(
                    features=["x3", "x4"],
                    exponents=[1.0, 1.0],
                    rhs=0.5,
                    sign=1,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is False

    def test_multiple_nchoosek_no_blockers(self):
        domain = Domain.from_lists(
            inputs=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(1, 5)],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                NChooseKConstraint(
                    features=["x3", "x4"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is True


class TestHasSemicontinuousFeatures:
    def test_no_continuous_inputs_with_allow_zero(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0.0, 1.0)),
                ContinuousInput(key="x2", bounds=(0.0, 1.0), allow_zero=False),
            ],
        )
        assert has_semicontinuous_features(domain) is False

    def test_allow_zero_with_positive_lb(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0.0, 1.0)),
                ContinuousInput(key="x2", bounds=(0.2, 1.0), allow_zero=True),
            ],
        )
        assert has_semicontinuous_features(domain) is True


class TestIsPruningApplicable:
    def test_neither_trigger(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
            ],
        )
        assert is_pruning_applicable(domain) is False

    def test_nchoosek_trigger(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
            ],
        )
        assert is_pruning_applicable(domain) is True

    def test_semicontinuous_trigger(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0.2, 1.0), allow_zero=True),
                ContinuousInput(key="x2", bounds=(0.0, 1.0)),
            ],
        )
        assert is_pruning_applicable(domain) is True

    def test_both_triggers(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0.2, 1.0), allow_zero=True),
                ContinuousInput(key="x2", bounds=(0.0, 1.0)),
                ContinuousInput(key="x3", bounds=(0.0, 1.0)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2", "x3"],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
            ],
        )
        assert is_pruning_applicable(domain) is True

    def test_semicontinuous_blocked_by_product(self):
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0.2, 1.0), allow_zero=True),
                ContinuousInput(key="x2", bounds=(0.0, 1.0)),
            ],
            constraints=[
                ProductInequalityConstraint(
                    features=["x1", "x2"],
                    exponents=[1.0, 1.0],
                    rhs=0.5,
                    sign=1,
                ),
            ],
        )
        # x1 (semicontinuous) is in a product constraint → not applicable
        assert is_pruning_applicable(domain) is False

    def test_nchoosek_blocked_overrides_semicontinuous(self):
        # Even though semi-continuous is present, the overall pruning is
        # blocked when NChooseK is itself blocked (defensive: caller should
        # not encounter this in practice).
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0.2, 1.0), allow_zero=True),
                ContinuousInput(key="x2", bounds=(0.0, 1.0)),
                ContinuousInput(key="x3", bounds=(0.0, 1.0)),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x2", "x3"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                ProductInequalityConstraint(
                    features=["x2", "x3"],
                    exponents=[1.0, 1.0],
                    rhs=0.5,
                    sign=1,
                ),
            ],
        )
        # NChooseK pruning blocked. But semi-continuous x1 is not in any
        # blocking constraint, so the semi-continuous path still allows
        # pruning.
        assert is_nchoosek_pruning_applicable(domain) is False
        assert is_pruning_applicable(domain) is True

    def test_unblocked_nchoosek_does_not_mask_semicontinuous_block(self):
        # Regression for a silent-gap: an unblocked NChooseK must not
        # cause the gate to skip the semi-continuous + blocking check.
        # x4 (semi-continuous) is in a ProductInequalityConstraint, so
        # pruning is unsafe and the gate must return False even though
        # the NChooseK over {x1, x2} is itself unblocked.
        domain = Domain.from_lists(
            inputs=[
                ContinuousInput(key="x1", bounds=(0.0, 1.0)),
                ContinuousInput(key="x2", bounds=(0.0, 1.0)),
                ContinuousInput(key="x3", bounds=(0.0, 1.0)),
                ContinuousInput(key="x4", bounds=(0.2, 1.0), allow_zero=True),
            ],
            constraints=[
                NChooseKConstraint(
                    features=["x1", "x2"],
                    min_count=0,
                    max_count=1,
                    none_also_valid=True,
                ),
                ProductInequalityConstraint(
                    features=["x3", "x4"],
                    exponents=[1.0, 1.0],
                    rhs=0.5,
                    sign=1,
                ),
            ],
        )
        assert is_nchoosek_pruning_applicable(domain) is True
        assert is_pruning_applicable(domain) is False
