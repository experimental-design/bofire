"""Shared utilities for dynamic Pydantic model registration."""

import typing
from collections.abc import Sequence
from typing import Optional, Union


def patch_field(model_cls: type, field_name: str, new_union: type) -> None:
    """Patch a Pydantic model field annotation with a new union type.

    Handles three annotation forms:
    - ``Union[A, B, ...]`` — replaced with *new_union*
    - ``Optional[Union[A, B, ...]]`` — wrapped as ``Optional[new_union]``
    - ``Sequence[Union[A, B, ...]]`` — wrapped as ``Sequence[new_union]``
    """
    old = model_cls.model_fields[  # ty: ignore[unresolved-attribute]
        field_name
    ].annotation
    args = typing.get_args(old)

    if not args:
        new = new_union
    elif type(None) in args:
        # Optional[X] is Union[X, None]
        new = Optional[new_union]
    elif typing.get_origin(old) in (list, Sequence):
        # Sequence[Union[...]] or list[Union[...]]
        new = Sequence[new_union]
    else:
        new = new_union

    model_cls.__annotations__[field_name] = new
    model_cls.model_fields[  # ty: ignore[unresolved-attribute]
        field_name
    ].annotation = new


def append_to_union_field(model_cls: type, field_name: str, new_type: type) -> None:
    """Append a type to the union inside a model field annotation.

    Detects the annotation structure (plain ``Union``, ``Optional[Union]``,
    or ``Sequence[Union]``) and appends *new_type* to the inner union.
    """
    old = model_cls.model_fields[  # ty: ignore[unresolved-attribute]
        field_name
    ].annotation
    origin = typing.get_origin(old)

    if origin in (list, Sequence):
        # Sequence[Union[A, B, ...]] → Sequence[Union[A, B, ..., new_type]]
        inner = typing.get_args(old)[0]
        inner_args = typing.get_args(inner)
        if new_type not in inner_args:
            new_inner = Union[tuple(list(inner_args) + [new_type])]
            new_ann = Sequence[new_inner]
            model_cls.__annotations__[field_name] = new_ann
            model_cls.model_fields[  # ty: ignore[unresolved-attribute]
                field_name
            ].annotation = new_ann
    else:
        # Union[A, B, ...] or Optional[Union[A, B, ...]]
        args = typing.get_args(old)
        has_none = type(None) in args
        non_none = [a for a in args if a is not type(None)]
        if new_type not in non_none:
            non_none.append(new_type)
            new_union = Union[tuple(non_none)]
            new_ann = Optional[new_union] if has_none else new_union
            model_cls.__annotations__[field_name] = new_ann
            model_cls.model_fields[  # ty: ignore[unresolved-attribute]
                field_name
            ].annotation = new_ann
