from inspect import Attribute
from typing import Any

import pytest
from bofire.domain.util import filter_by_class


class A:
    pass


class A1(A):
    pass


class A11(A1):
    pass


class A12(A1):
    pass


class A2(A):
    pass


class A21(A2):
    pass


class A211(A21):
    pass


class B:
    pass


class B1(B):
    pass


class B11(B1):
    pass


class B12(B1):
    pass


class B2(B):
    pass


class B21(B2):
    pass


class B211(B21):
    pass


class C:
    def __init__(self, attribute: Any) -> None:
        self.attribute = attribute


a = A()
a1 = A1()
a11 = A11()
a12 = A12()
a2 = A2()
a21 = A21()
a211 = A211()

b = B()
b1 = B1()
b11 = B11()
b12 = B12()
b2 = B2()
b21 = B21()
b211 = B211()


c = C(attribute=b)
c1 = C(attribute=b1)
c11 = C(attribute=b11)
c12 = C(attribute=b12)
c2 = C(attribute=b2)
c21 = C(attribute=b21)
c211 = C(attribute=b211)
c3 = C(attribute=None)

data = [
    a,
    a1,
    a11,
    a12,
    a2,
    a21,
    a211,
]

data2 = [
    a,
    b,
    c,
    c1,
    c11,
    c12,
    c2,
    c21,
    c211,
    c3,
]


@pytest.mark.parametrize(
    "data,includes,expected",
    [
        (data, [A], data),
        (data, [A, A1, A21], data),
        (data, [A1], [a1, a11, a12]),
        (data, [A11], [a11]),
        (data, [A2], [a2, a21, a211]),
    ],
)
def test_filter_by_class(data, includes, expected):
    res = filter_by_class(data, includes)
    print("got:", [type(x).__name__ for x in res])
    print("expected:", [type(x).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,by_attribute,expected",
    [
        (data2, [B], "attribute", [c, c1, c11, c12, c2, c21, c211]),
        (data2, [B], "attribute2", []),
        (data, [B], "attribute", []),
        (data, [B], "attribute", []),
        (data2, [B, B1, B2], "attribute", [c, c1, c11, c12, c2, c21, c211]),
        (data2, [B1], "attribute", [c1, c11, c12]),
        (data2, [B11], "attribute", [c11]),
        (data2, [B2], "attribute", [c2, c21, c211]),
    ],
)
def test_filter_by_class_attribute(data, includes, by_attribute, expected):
    res = filter_by_class(data, includes, by_attribute=by_attribute)
    print("got:", [type(getattr(x, by_attribute)).__name__ for x in res])
    print("expected:", [type(getattr(x, by_attribute)).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,expected",
    [
        (data, [A], [a]),
        (data, [A, A1, A21], [a, a1, a21]),
        (data, [A1], [a1]),
        (data, [A11], [a11]),
        (data, [A2], [a2]),
    ],
)
def test_filter_by_class_exact(data, includes, expected):
    res = filter_by_class(data, includes, exact=True)
    print("got:", [type(x).__name__ for x in res])
    print("expected:", [type(x).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,by_attribute,expected",
    [
        (data2, [B], "attribute", [c]),
        (data2, [B, B1, B21], "attribute", [c, c1, c21]),
        (data2, [B1], "attribute", [c1]),
        (data2, [B11], "attribute", [c11]),
        (data2, [B2], "attribute", [c2]),
    ],
)
def test_filter_by_class_attribute_exact(data, includes, by_attribute, expected):
    res = filter_by_class(data, includes, exact=True, by_attribute=by_attribute)
    print("got:", [type(getattr(x, by_attribute)).__name__ for x in res])
    print("expected:", [type(getattr(x, by_attribute)).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,exclude,expected",
    [
        (data, [A], [], data),
        (data, [A1], [A11], [a1, a12]),
        (data, [A], [A21], [a, a1, a11, a12, a2]),
        (data, [A2], [A1], [a2, a21, a211]),
    ],
)
def test_filter_by_class_exclude(data, includes, exclude, expected):
    res = filter_by_class(data, includes, exclude)
    print("got:", [type(x).__name__ for x in res])
    print("expected:", [type(x).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,by_attribute,exclude,expected",
    [
        (data2, [B], "attribute", [], [c, c1, c11, c12, c2, c21, c211]),
        (data2, [B1], "attribute", [B11], [c1, c12]),
        (data2, [B], "attribute", [B21], [c, c1, c11, c12, c2]),
        (data2, [B2], "attribute", [B1], [c2, c21, c211]),
    ],
)
def test_filter_by_class_exclude_attribute(
    data, includes, by_attribute, exclude, expected
):
    res = filter_by_class(data, includes, exclude, by_attribute=by_attribute)
    print("got:", [type(getattr(x, by_attribute)).__name__ for x in res])
    print("expected:", [type(getattr(x, by_attribute)).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,exclude,expected",
    [
        (data, [A], [], [a]),
        (data, [A1], [A11], [a1]),
        (data, [A1], [], [a1]),
        (data, [A1, A11], [], [a1, a11]),
        (data, [A, A2, A21], [A211], [a, a2, a21]),
    ],
)
def test_filter_by_class_exclude_exact(data, includes, exclude, expected):
    res = filter_by_class(data, includes, exclude, exact=True)
    print("got:", [type(x).__name__ for x in res])
    print("expected:", [type(x).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,by_attribute,exclude,expected",
    [
        (data2, [B], "attribute", [], [c]),
        (data2, [B1], "attribute", [B11], [c1]),
        (data2, [B1], "attribute", [], [c1]),
        (data2, [B1, B11], "attribute", [], [c1, c11]),
        (data2, [B, B2, B21], "attribute", [B211], [c, c2, c21]),
    ],
)
def test_filter_by_class_attribute_exclude_exact(
    data, includes, by_attribute, exclude, expected
):
    res = filter_by_class(
        data, includes, exclude, exact=True, by_attribute=by_attribute
    )
    print("got:", [type(getattr(x, by_attribute)).__name__ for x in res])
    print("expected:", [type(getattr(x, by_attribute)).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "includes",
    [
        ([]),
    ],
)
def test_filter_by_class_no_cls(includes):
    with pytest.raises(ValueError):
        filter_by_class(data, includes)


@pytest.mark.parametrize(
    "includes, excludes",
    [
        ([A], [A]),
        ([A2], [A2]),
        ([A12], [A, A12]),
        ([A11, A21], [A12, A21]),
    ],
)
def test_filter_by_class_no_intersection(includes, excludes):
    with pytest.raises(ValueError):
        filter_by_class(data, includes, excludes)
