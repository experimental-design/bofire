from typing import Union

import pytest
from pydantic import BaseModel

from bofire.data_models import unions


class A(BaseModel):
    pass


class B(BaseModel):
    pass


class C(BaseModel):
    pass


ABC = Union[A, B, C]
AB = Union[A, B]
AC = Union[A, C]


@pytest.mark.parametrize(
    "union, expected",
    [
        (ABC, {A, B, C}),
        (AB, {A, B}),
        (AC, {A, C}),
    ],
)
def test_union_to_list_should_process_union(union, expected):
    assert set(unions.to_list(union)) == expected


@pytest.mark.parametrize(
    "single",
    [
        A,
        B,
        C,
    ],
)
def test_union_to_list_should_process_single_model(single):
    assert unions.to_list(single) == [single]
