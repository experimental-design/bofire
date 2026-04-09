from bofire.data_models.constraints.api import (
    ProductEqualityConstraint,
    ProductInequalityConstraint,
)


def test_product_equality_to_description():
    c = ProductEqualityConstraint(
        features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1
    )
    assert c.to_description() == "x1^2.0 * x2^3.0 = 1.0"


def test_product_inequality_to_description():
    c = ProductInequalityConstraint(
        features=["x1", "x2"], exponents=[2, 3], rhs=1.0, sign=1
    )
    assert c.to_description() == "x1^2.0 * x2^3.0 <= 1.0"
