from typing import Type, Union, _UnionGenericAlias  # type: ignore


def to_list(union: Union[Type, _UnionGenericAlias]):
    if isinstance(union, Type):
        return [union]
    if isinstance(union, _UnionGenericAlias):
        return union.__args__
    raise TypeError(
        f"expected argument of type Type or _UnionGenericAlias, got {type(union)}"
    )
