from bofire.data_models.constraints.api import (
    ProductEqualityConstraint,
    ProductInequalityConstraint,
)


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
