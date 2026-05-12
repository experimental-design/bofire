"""Shared utilities for dynamic Pydantic model registration."""

import typing
from collections.abc import Sequence
from typing import Optional, Tuple, Type

from bofire.data_models.unions import discriminator_name, tagged_union, unwrap_annotated


def _rewrap_union(types: Tuple[Type, ...], discriminator: Optional[str]):
    """Build a (possibly tagged) union from *types*."""
    if discriminator is None:
        return typing.Union[types]
    return tagged_union(*types, discriminator=discriminator)


def _set_field_annotation(model_cls: type, field_name: str, annotation) -> None:
    """Write *annotation* to both ``__annotations__`` and ``model_fields[...]``."""
    model_cls.__annotations__[field_name] = annotation
    model_cls.model_fields[  # ty: ignore[unresolved-attribute]
        field_name
    ].annotation = annotation


def patch_field(model_cls: type, field_name: str, new_union: type) -> None:
    """Patch a Pydantic model field annotation with a new union type.

    Handles three annotation forms (plus any ``Annotated[...]`` wrapper):
    - ``Union[A, B, ...]`` — replaced with *new_union*
    - ``Optional[Union[A, B, ...]]`` — wrapped as ``Optional[new_union]``
    - ``Sequence[Union[A, B, ...]]`` — wrapped as ``Sequence[new_union]``
    """
    old = model_cls.model_fields[  # ty: ignore[unresolved-attribute]
        field_name
    ].annotation
    old_unwrapped, _ = unwrap_annotated(old)
    args = typing.get_args(old_unwrapped)

    if not args:
        new = new_union
    elif type(None) in args:
        # Optional[X] is Union[X, None]
        new = typing.Optional[new_union]
    elif typing.get_origin(old_unwrapped) in (list, Sequence):
        # Sequence[Union[...]] or list[Union[...]]
        new = Sequence[new_union]
    else:
        new = new_union

    _set_field_annotation(model_cls, field_name, new)


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
        inner_unwrapped, inner_meta = unwrap_annotated(inner)
        discriminator = discriminator_name(inner_meta)
        inner_args = typing.get_args(inner_unwrapped)
        if new_type not in inner_args:
            new_inner = _rewrap_union((*inner_args, new_type), discriminator)
            _set_field_annotation(model_cls, field_name, Sequence[new_inner])
    else:
        # Union[...] / Optional[Union[...]] / Annotated[Union[...], Field(...)]
        old_unwrapped, meta = unwrap_annotated(old)
        discriminator = discriminator_name(meta)
        args = typing.get_args(old_unwrapped)
        has_none = type(None) in args
        non_none = tuple(a for a in args if a is not type(None))
        if new_type not in non_none:
            new_union = _rewrap_union((*non_none, new_type), discriminator)
            new_ann = typing.Optional[new_union] if has_none else new_union
            _set_field_annotation(model_cls, field_name, new_ann)
