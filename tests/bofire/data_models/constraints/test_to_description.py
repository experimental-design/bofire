"""Tests for Constraint.to_description() on all constraint types."""

from bofire.data_models.constraints.api import (
    NChooseKConstraint,
    ProductEqualityConstraint,
    ProductInequalityConstraint,
)


def test_nchoosek_to_description():
    c = NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    desc = c.to_description()
    assert "Choose 1-2" in desc
    assert "x1" in desc


def test_nchoosek_to_description_none_valid():
    c = NChooseKConstraint(
        features=["x1", "x2"],
        min_count=1,
        max_count=2,
        none_also_valid=True,
    )
    assert "or none" in c.to_description()


def test_product_equality_to_description():
    c = ProductEqualityConstraint(
        features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1
    )
    desc = c.to_description()
    assert "x1^2" in desc
    assert "= 1.0" in desc


def test_product_inequality_to_description():
    c = ProductInequalityConstraint(
        features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1
    )
    assert "<= 1.0" in c.to_description()
