"""Tests for the domain-level pruning applicability gates.

These functions live in `bofire.strategies.predictives._nchoosek_pruning`
as free functions (moved out of `Domain` in Phase 2 of the BONSAI
pruning refactor).
"""

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput
from bofire.strategies.predictives._nchoosek_pruning import (
    has_nchoosek_linear_overlap,
    has_semicontinuous_features,
    is_nchoosek_pruning_applicable,
    is_pruning_applicable,
)


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
