import pytest
from everest.domain.util import filter_by_class


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


a = A()
a1 = A1()
a11 = A11()
a12 = A12()
a2 = A2()
a21 = A21()
a211 = A211()

data = [
    a, a1, a11, a12, a2, a21, a211,
]


@pytest.mark.parametrize(
    "data,includes,expected",
    [
        (data, [A], data),
        (data, [A, A1, A21], data),
        (data, [A1], [a1, a11, a12]),
        (data, [A11], [a11]),
        (data, [A2], [a2, a21, a211]),
    ]
)
def test_filter_by_class(data, includes, expected):
    res = filter_by_class(data, includes)
    print("got:", [type(x).__name__ for x in res])
    print("expected:", [type(x).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,expected",
    [
        (data, [A], [a]),
        (data, [A, A1, A21], [a, a1, a21]),
        (data, [A1], [a1]),
        (data, [A11], [a11]),
        (data, [A2], [a2]),
    ]
)
def test_filter_by_class_exact(data, includes, expected):
    res = filter_by_class(data, includes, exact=True)
    print("got:", [type(x).__name__ for x in res])
    print("expected:", [type(x).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,exclude,expected",
    [
        (data, [A], [], data),
        (data, [A1], [A11], [a1, a12]),
        (data, [A], [A21], [a, a1, a11, a12, a2]),
        (data, [A2], [A1], [a2, a21, a211]),
    ]
)
def test_filter_by_class_exclude(data, includes, exclude, expected):
    res = filter_by_class(data, includes, exclude)
    print("got:", [type(x).__name__ for x in res])
    print("expected:", [type(x).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "data,includes,exclude,expected",
    [
        (data, [A], [], [a]),
        (data, [A1], [A11], [a1]),
        (data, [A1], [], [a1]),
        (data, [A1, A11], [], [a1, a11]),
        (data, [A, A2, A21], [A211], [a, a2, a21]),
    ]
)
def test_filter_by_class_exclude_exact(data, includes, exclude, expected):
    res = filter_by_class(data, includes, exclude, exact=True)
    print("got:", [type(x).__name__ for x in res])
    print("expected:", [type(x).__name__ for x in expected])
    assert set(res) == set(expected)


@pytest.mark.parametrize(
    "includes",
    [
        ([]),
    ]
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
    ]
)
def test_filter_by_class_no_intersection(includes, excludes):
    with pytest.raises(ValueError):
        filter_by_class(data, includes, excludes)
