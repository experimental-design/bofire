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
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.strategies.predictives import _nchoosek_pruning as ncp
from bofire.strategies.predictives._nchoosek_pruning import (
    PruningInfeasibleError,
    _active_counts,
    _build_active_variant,
    _build_zero_variant,
    _classify_features_for_row,
    _features_eligible_for_zero,
    _is_nchoosek_fulfilled,
    _violated_constraints,
    _zero_action_blocked_by_min_count,
    prune_nchoosek,
)
from bofire.strategies.utils import get_torch_bounds_from_domain
from bofire.utils.torch_tools import get_linear_constraints, tkwargs


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
    """AF returning a constant for every input — exercises the
    ``dense_incremental == 0`` branch of ``_af_reduction``.
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
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3)) is False

    def test_fulfilled_min_count_violated(self):
        c = self._basic_constraint(min_count=2, max_count=3)
        x = _row_to_tensor([0.5, 0.0, 0.0])
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3)) is False

    def test_fulfilled_none_also_valid_zero_count(self):
        c = self._basic_constraint(min_count=1, max_count=2, none_also_valid=True)
        x = _row_to_tensor([0.0, 0.0, 0.0])
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3)) is True

    def test_fulfilled_none_also_valid_does_not_allow_low_nonzero(self):
        c = self._basic_constraint(min_count=2, max_count=3, none_also_valid=True)
        x = _row_to_tensor([0.5, 0.0, 0.0])  # one active, below min_count
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3)) is False

    def test_fulfilled_tol_classifies_small_values_as_zero(self):
        c = self._basic_constraint(min_count=1, max_count=2)
        x = _row_to_tensor([1e-9, 1e-9, 0.5])
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3)) is True

    def test_fulfilled_within_bounds(self):
        c = self._basic_constraint(min_count=1, max_count=2)
        x = _row_to_tensor([0.5, 0.5, 0.0])
        assert _is_nchoosek_fulfilled(x, [c], self._f2i_continuous(3)) is True

    # ----- _active_counts -----

    def test_active_counts_per_constraint(self):
        c1 = self._basic_constraint(features=["x1", "x2"], max_count=1)
        c2 = self._basic_constraint(features=["x2", "x3", "x4"], max_count=2)
        x = _row_to_tensor([0.5, 0.5, 0.5, 0.0])
        f2i = self._f2i_continuous(4)
        counts = _active_counts(x, [c1, c2], f2i, tol=1e-6)
        assert counts == {0: 2, 1: 2}

    # ----- _violated_constraints -----

    def test_violated_constraints_only_max_violations(self):
        c1 = self._basic_constraint(features=["x1", "x2"], max_count=1, min_count=0)
        c2 = self._basic_constraint(features=["x3", "x4"], max_count=2, min_count=2)
        counts = {0: 2, 1: 1}  # c1 violates max, c2 violates min (but excluded)
        violated = _violated_constraints(counts, [c1, c2])
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
        variant, valid = _build_zero_variant(
            x,
            j_idx=2,
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
        variant, valid = _build_zero_variant(
            x,
            j_idx=2,
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
        variant, valid = _build_zero_variant(
            x,
            j_idx=0,
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
        variant, valid = _build_zero_variant(
            x,
            j_idx=0,
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
        variant, valid = _build_active_variant(
            x,
            j_idx=2,
            lb_j=0.2,
            ub_j=1.0,
            bounds=inp["bounds"],
            inequality_constraints=inp["inequality_constraints"],
            equality_constraints=inp["equality_constraints"],
            acqf=self._acqf(),
            per_step_local_reopt=False,
        )
        assert valid is True
        assert variant[2].item() >= 0.2 - 1e-6
        assert variant.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_active_variant_no_constraints_keeps_starting_or_lb(self):
        # No linear constraints — projection short-circuits, we just clamp
        # the fractional column to lb.
        x = _row_to_tensor([0.5, 0.05, 0.5, 0.5])
        variant, valid = _build_active_variant(
            x,
            j_idx=1,
            lb_j=0.2,
            ub_j=1.0,
            bounds=torch.tensor([[0.0] * 4, [1.0] * 4], **tkwargs),
            inequality_constraints=[],
            equality_constraints=[],
            acqf=self._acqf(),
            per_step_local_reopt=False,
        )
        assert valid is True
        assert variant[1].item() == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# End-to-end pruning tests
# ---------------------------------------------------------------------------


class TestPruneNchoosekEndToEnd:
    # ----- Single NChooseK without semi-continuity (matches today's path) -----

    def test_q1_max_count_violation_zeros_smallest_weight(self):
        domain = _make_simple_domain(
            n_features=4,
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
        domain = _make_simple_domain(
            n_features=4,
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
        domain = _make_simple_domain(
            n_features=4,
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

    def test_dense_incremental_zero_branch(self):
        # ConstantAcqf returns the same value for every input → no
        # exception, deterministic action selection.
        domain = _make_simple_domain(
            n_features=4,
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
        domain = _make_simple_domain(
            n_features=4,
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

    # ----- Caller-supplied fixed_features -----

    def test_fixed_feature_outside_nchoosek_does_not_move(self):
        # 5-feature domain: x5 is fixed at 0.7 by the caller. NChooseK
        # acts only on x1..x4. Mixture x1+...+x4 = 1 (does not include
        # the fixed feature). The pruning should not touch x5.
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
        X = _stack_to_tensor([[0.4, 0.3, 0.2, 0.1, 0.7]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            fixed_features={4: 0.7},
            **inp,
            final_local_reopt=True,
        )
        assert out[0, 4].item() == pytest.approx(0.7, abs=1e-6)
        nz = (out[0, :4].abs() > 1e-6).sum().item()
        assert nz <= 2
        assert out[0, :4].sum().item() == pytest.approx(1.0, abs=1e-3)

    def test_fixed_zero_feature_does_not_count_or_move(self):
        # x5 is fixed to zero by the caller — it should never be a
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
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5, 0.0]])
        out = prune_nchoosek(
            X=X,
            acqf=acqf,
            fixed_features={4: 0.0},
            **inp,
            final_local_reopt=True,
        )
        assert out[0, 4].item() == pytest.approx(0.0)
        nz = (out[0, :4].abs() > 1e-6).sum().item()
        assert nz <= 2

    def test_fixed_feature_inside_nchoosek_excluded_from_actions(self):
        # x1 is fixed at 0.5 even though it appears in the NChooseK
        # constraint. The pruning must exclude it from the action set
        # (cannot zero a fixed feature) and instead zero among
        # {x2, x3, x4}.
        domain = _make_simple_domain(
            n_features=4,
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
        # Use weights that would otherwise make x1 the prime zero
        # candidate (smallest weight). The fixed-features filter must
        # override.
        weights = torch.tensor([0.1, 5.0, 5.0, 10.0], **tkwargs)
        acqf = WeightedSumAcqf(weights)
        X = _stack_to_tensor([[0.5, 0.5, 0.5, 0.5]])
        out = prune_nchoosek(
            X=X,
            acqf=cast(AcquisitionFunction, acqf),
            fixed_features={0: 0.5},
            **inp,
            final_local_reopt=False,
        )
        assert out[0, 0].item() == pytest.approx(0.5)
        nz = (out[0].abs() > 1e-6).sum().item()
        assert nz <= 2
