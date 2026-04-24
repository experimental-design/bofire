from typing import Annotated, Optional, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, Field

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


def test_to_list_unwraps_annotated_tagged_union():
    tagged = unions.tagged_union(A, B, C)
    assert set(unions.to_list(tagged)) == {A, B, C}


def test_tagged_union_uses_type_discriminator_by_default():
    tagged = unions.tagged_union(A, B)
    unwrapped, meta = unions.unwrap_annotated(tagged)
    assert get_origin(unwrapped) is Union
    assert set(get_args(unwrapped)) == {A, B}
    assert unions.discriminator_name(meta) == "type"


def test_tagged_union_accepts_custom_discriminator():
    tagged = unions.tagged_union(A, B, discriminator="kind")
    _, meta = unions.unwrap_annotated(tagged)
    assert unions.discriminator_name(meta) == "kind"


def test_unwrap_annotated_passthrough_on_bare_type():
    unwrapped, meta = unions.unwrap_annotated(A)
    assert unwrapped is A
    assert meta == ()


def test_unwrap_annotated_returns_inner_and_metadata():
    annotated = Annotated[A, "tag", 42]
    inner, meta = unions.unwrap_annotated(annotated)
    assert inner is A
    assert meta == ("tag", 42)


def test_discriminator_name_returns_none_when_absent():
    assert unions.discriminator_name(("other", 1)) is None
    assert unions.discriminator_name((Field(default=0),)) is None


def test_extract_union_args_returns_types_and_discriminator():
    args, discriminator = unions.extract_union_args(unions.tagged_union(A, B))
    assert set(args) == {A, B}
    assert discriminator == "type"


def test_extract_union_args_handles_plain_union():
    args, discriminator = unions.extract_union_args(ABC)
    assert set(args) == {A, B, C}
    assert discriminator is None


def test_extract_union_args_wraps_single_type_as_one_tuple():
    args, discriminator = unions.extract_union_args(A)
    assert args == (A,)
    assert discriminator is None


def test_extract_union_args_preserves_none_for_optional():
    args, _ = unions.extract_union_args(Optional[A])
    assert set(args) == {A, type(None)}
