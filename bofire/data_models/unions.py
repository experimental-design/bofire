from typing import Annotated, Type, Union, get_args, get_origin


def unwrap_annotated(tp):
    """Return ``(inner, metadata)``. If *tp* is not ``Annotated``, metadata is ``()``."""
    if get_origin(tp) is Annotated:
        args = get_args(tp)
        return args[0], args[1:]
    return tp, ()


def to_list(union: Type):
    # Unwrap Annotated[Union[...], Field(discriminator="type")] to reach the
    # underlying Union before extracting its arguments.
    union, _ = unwrap_annotated(union)
    if get_origin(union) is Union:
        return get_args(union)
    return [union]
