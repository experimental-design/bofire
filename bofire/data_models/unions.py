from typing import Annotated, Optional, Tuple, Type, Union, get_args, get_origin

from pydantic import Field


def unwrap_annotated(tp):
    """Return ``(inner, metadata)``. If *tp* is not ``Annotated``, metadata is ``()``."""
    if get_origin(tp) is Annotated:
        args = get_args(tp)
        return args[0], args[1:]
    return tp, ()


def discriminator_name(metadata) -> Optional[str]:
    """Return the discriminator field name if *metadata* contains a Field with one."""
    for meta in metadata:
        if isinstance(meta, type(Field())) and getattr(meta, "discriminator", None):
            return meta.discriminator
    return None


def extract_union_args(tp) -> Tuple[Tuple[Type, ...], Optional[str]]:
    """Return ``(union_args, discriminator)`` for a union-like annotation.

    Unwraps ``Annotated[...]`` wrappers and handles bare types (returned as a
    one-tuple). The discriminator is ``None`` unless the annotation is wrapped
    with ``Field(discriminator=...)``.
    """
    unwrapped, meta = unwrap_annotated(tp)
    discriminator = discriminator_name(meta)
    if get_origin(unwrapped) is Union:
        return get_args(unwrapped), discriminator
    return (unwrapped,), discriminator


def to_list(union: Type):
    union, _ = unwrap_annotated(union)
    if get_origin(union) is Union:
        return get_args(union)
    return [union]


def tagged_union(*types: Type, discriminator: str = "type") -> Type:
    """Build ``Annotated[Union[*types], Field(discriminator=discriminator)]``.

    Convenience wrapper for the tagged-union pattern used across ``api.py``
    modules. Accepts types as varargs; for dynamic lists, splat:
    ``tagged_union(*_TYPES)``.
    """
    return Annotated[Union[types], Field(discriminator=discriminator)]
