"""Shared utilities for dynamic Pydantic model registration."""

import typing
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple, Type

from pydantic_core import PydanticUndefined

from bofire.data_models.unions import discriminator_name, tagged_union, unwrap_annotated


def discriminator_value(cls, discriminator: str = "type"):
    """Return the discriminator literal value of a data model class.

    For a field declared as ``type: Literal["Foo"] = "Foo"`` this returns
    ``"Foo"``. Returns ``None`` if the class has no such field or it has no
    concrete default (e.g. an abstract base class).
    """
    fields = getattr(cls, "model_fields", None)
    if not fields:
        return None
    field = fields.get(discriminator)
    if field is None:
        return None
    default = getattr(field, "default", None)
    if default is PydanticUndefined:
        return None
    return default


def find_registered_conflict(registry, data_model_cls, discriminator: str = "type"):
    """Return an already-registered class that shares *data_model_cls*'s
    discriminator value but is a *different* class object, or ``None``.

    This is the situation that produces the cryptic Pydantic error
    ``Value '...' for discriminator '...' mapped to multiple choices`` once a
    dependent discriminated union is built.
    """
    value = discriminator_value(data_model_cls, discriminator)
    if value is None:
        return None
    for existing in registry:
        if existing is data_model_cls:
            continue
        if discriminator_value(existing, discriminator) == value:
            return existing
    return None


def register_into(
    registry: List[type],
    data_model_cls: type,
    *,
    overwrite: bool = False,
    kind: str = "type",
    discriminator: str = "type",
) -> Tuple[str, Optional[type]]:
    """Insert *data_model_cls* into the *registry* list in place.

    Handles three cases:

    - the exact class is already registered -> ``("noop", None)`` (idempotent)
    - no conflict -> appends and returns ``("add", None)``
    - a *different* class with the same discriminator value is registered ->
      if *overwrite* is ``True`` the old class is replaced in place and
      ``("replace", old_cls)`` is returned, otherwise a ``ValueError`` is
      raised describing the conflict.

    Args:
        registry: Mutable list of registered data model classes.
        data_model_cls: The class to register.
        overwrite: Replace an existing same-discriminator class instead of
            raising.
        kind: Human readable noun used in the error message (e.g. ``"strategy"``).
        discriminator: The discriminator field name (defaults to ``"type"``).

    Raises:
        ValueError: On a discriminator collision when *overwrite* is ``False``.
    """
    if data_model_cls in registry:
        return ("noop", None)

    conflict = find_registered_conflict(registry, data_model_cls, discriminator)
    if conflict is None:
        registry.append(data_model_cls)
        return ("add", None)

    if not overwrite:
        value = discriminator_value(data_model_cls, discriminator)
        raise ValueError(
            f"A {kind} with {discriminator}={value!r} is already registered as "
            f"'{conflict.__module__}.{conflict.__qualname__}'. This usually "
            f"happens when the same {kind} is defined and registered more than "
            f"once (for example by re-running a notebook cell or import). Pass "
            f"overwrite=True to replace the existing registration, or give the "
            f"new {kind} a distinct '{discriminator}' value."
        )

    registry[registry.index(conflict)] = data_model_cls
    return ("replace", conflict)


def swap_or_append(registry: List[type], new_cls: type, replaced: Optional[type]):
    """Mirror a registration onto a secondary registry list.

    Replaces *replaced* with *new_cls* if present, otherwise appends *new_cls*
    if not already there. Used to keep sub-category registries (e.g.
    continuous/categorical kernels) in sync with the main registry.
    """
    if replaced is not None and replaced in registry:
        registry[registry.index(replaced)] = new_cls
    elif new_cls not in registry:
        registry.append(new_cls)


def pop_conflicting_map_keys(
    map_dict: Dict[type, object], data_model_cls: type, discriminator: str = "type"
) -> None:
    """Drop entries from a mapper dict whose key shares *data_model_cls*'s
    discriminator value but is a different (stale) class object.

    Keeps functional-mapper dicts from accumulating dead entries when a type is
    re-registered with ``overwrite=True``.
    """
    value = discriminator_value(data_model_cls, discriminator)
    if value is None:
        return
    stale = [
        key
        for key in map_dict
        if key is not data_model_cls
        and discriminator_value(key, discriminator) == value
    ]
    for key in stale:
        del map_dict[key]


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
        inner_args = _drop_discriminator_conflicts(
            typing.get_args(inner_unwrapped), new_type, discriminator
        )
        if new_type not in inner_args:
            new_inner = _rewrap_union((*inner_args, new_type), discriminator)
            _set_field_annotation(model_cls, field_name, Sequence[new_inner])
    else:
        # Union[...] / Optional[Union[...]] / Annotated[Union[...], Field(...)]
        old_unwrapped, meta = unwrap_annotated(old)
        discriminator = discriminator_name(meta)
        args = typing.get_args(old_unwrapped)
        has_none = type(None) in args
        non_none = _drop_discriminator_conflicts(
            tuple(a for a in args if a is not type(None)), new_type, discriminator
        )
        if new_type not in non_none:
            new_union = _rewrap_union((*non_none, new_type), discriminator)
            new_ann = typing.Optional[new_union] if has_none else new_union
            _set_field_annotation(model_cls, field_name, new_ann)


def _drop_discriminator_conflicts(
    members: Tuple[type, ...], new_type: type, discriminator: Optional[str]
) -> Tuple[type, ...]:
    """Remove members sharing *new_type*'s discriminator value but a different
    identity, so re-registering a type (``overwrite=True``) replaces the stale
    member in an inline union instead of producing a duplicate-tag union.
    """
    field = discriminator or "type"
    new_value = discriminator_value(new_type, field)
    if new_value is None:
        return members
    return tuple(
        m
        for m in members
        if m is new_type or discriminator_value(m, field) != new_value
    )
