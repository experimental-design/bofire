from typing import Annotated, Type, Union, get_args, get_origin


def to_list(union: Type):
    # Unwrap Annotated[Union[...], Field(discriminator="type")] to reach the
    # underlying Union before extracting its arguments.
    if get_origin(union) is Annotated:
        union = get_args(union)[0]
    if get_origin(union) is Union:
        return get_args(union)
    return [union]
