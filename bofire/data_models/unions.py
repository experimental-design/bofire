from typing import Type, Union, get_args, get_origin


def to_list(union: Type):
    if get_origin(union) is Union:
        return get_args(union)
    return [union]
