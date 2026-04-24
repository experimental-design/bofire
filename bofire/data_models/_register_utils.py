"""Shared utilities for dynamic Pydantic model registration."""

import typing
from collections.abc import Sequence
from typing import Annotated, Optional, Union

from pydantic import Field


def _unwrap_annotated(tp):
    """Return (inner, metadata). If *tp* is not Annotated, metadata is ()."""
    if typing.get_origin(tp) is Annotated:
        args = typing.get_args(tp)
        return args[0], args[1:]
    return tp, ()


def _discriminator_name(metadata):
    """Return the discriminator field name if *metadata* contains a Field with one."""
    for meta in metadata:
        if isinstance(meta, type(Field())) and getattr(meta, "discriminator", None):
            return meta.discriminator
    return None


def _rewrap_union(union_tp, discriminator: Optional[str]):
    """Wrap *union_tp* in Annotated[..., Field(discriminator=...)] if requested."""
    if discriminator is None:
        return union_tp
    return Annotated[union_tp, Field(discriminator=discriminator)]


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
    ``Sequence[Union]``, or any of these wrapped in ``Annotated[..., Field(
    discriminator=...)]``) and appends *new_type* to the inner union,
    preserving the discriminator tag if present.
    """
    old = model_cls.model_fields[  # ty: ignore[unresolved-attribute]
        field_name
    ].annotation
    origin = typing.get_origin(old)

    if origin in (list, Sequence):
        # Sequence[Union[...]] or Sequence[Annotated[Union[...], Field(...)]]
        inner = typing.get_args(old)[0]
        inner_unwrapped, inner_meta = _unwrap_annotated(inner)
        discriminator = _discriminator_name(inner_meta)
        inner_args = typing.get_args(inner_unwrapped)
        if new_type not in inner_args:
            new_inner_union = Union[tuple(list(inner_args) + [new_type])]
            new_inner = _rewrap_union(new_inner_union, discriminator)
            new_ann = Sequence[new_inner]
            model_cls.__annotations__[field_name] = new_ann
            model_cls.model_fields[  # ty: ignore[unresolved-attribute]
                field_name
            ].annotation = new_ann
    else:
        # Union[...] / Optional[Union[...]] / Annotated[Union[...], Field(...)]
        old_unwrapped, meta = _unwrap_annotated(old)
        discriminator = _discriminator_name(meta)
        args = typing.get_args(old_unwrapped)
        has_none = type(None) in args
        non_none = [a for a in args if a is not type(None)]
        if new_type not in non_none:
            non_none.append(new_type)
            new_union = Union[tuple(non_none)]
            new_union = _rewrap_union(new_union, discriminator)
            new_ann = Optional[new_union] if has_none else new_union
            model_cls.__annotations__[field_name] = new_ann
            model_cls.model_fields[  # ty: ignore[unresolved-attribute]
                field_name
            ].annotation = new_ann
