import collections.abc as collections
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Type, Union, get_args, get_origin


def filter_by_attribute(
    data: Sequence,
    attribute_getter: Callable[[Type], Any],
    includes: Optional[Union[Type, Sequence[Type]]] = None,
    excludes: Optional[Union[Type, Sequence[Type]]] = None,
    exact: bool = False,
) -> List:
    """Returns those data elements where the attribute is of one of the include types.

    Args:
        data: to be filtered
        attribute_getter: expects an item of the data list and returns the attribute to filter by
        includes: attribute types that should be kept, sub-type are included by default, see exact
        excludes: attribute types that will be excluded even if they are sub-types of or include types.
        exact: true for not including subtypes

    Returns:
        list of data point with attributes as filtered for

    """
    data_with_attr = []
    for d in data:
        try:
            attribute_getter(d)
            data_with_attr.append(d)
        except AttributeError:
            pass

    filtered = filter_by_class(
        data_with_attr,
        includes=includes,
        excludes=excludes,
        exact=exact,
        key=attribute_getter,
    )
    return filtered


def filter_by_class(
    data: Sequence,
    includes: Optional[Union[Type, Sequence[Type]]] = None,
    excludes: Optional[Union[Type, Sequence[Type]]] = None,
    exact: bool = False,
    key: Callable[[Type], Any] = lambda x: x,
) -> List:
    """Returns those data elements where are one of the include types.

    Args:
        data: to be filtered
        includes: attribute types that should be kept, sub-type are included by default, see exact
        excludes: attribute types that will be excluded even if they are sub-types of or include types.
        exact: true for not including subtypes
        key: maps a data list item to something that is used for filtering, identity by default

    Returns:
        filtered list of data points

    """
    if includes is None:
        includes = []
    if not isinstance(includes, collections.Sequence):
        includes = [includes]
    if excludes is None:
        excludes = []
    if not isinstance(excludes, collections.Sequence):
        excludes = [excludes]

    if len(includes) == len(excludes) == 0:
        raise ValueError("no filter provided")

    if len(includes) == 0:
        includes = [object]

    if len([x for x in includes if x in excludes]) > 0:
        raise ValueError("includes and excludes overlap")

    includes_ = []
    for incl in includes:
        if get_origin(incl) is Union:
            includes_ += get_args(incl)
        else:
            includes_.append(incl)
    includes = includes_
    excludes_ = []
    for excl in excludes:
        if get_origin(excl) is Union:
            excludes_ += get_args(excl)
        else:
            excludes_.append(excl)
    excludes = excludes_

    if exact:
        return [
            d for d in data if type(key(d)) in includes and type(key(d)) not in excludes
        ]
    return [
        d
        for d in data
        if isinstance(key(d), tuple(includes))  # type: ignore
        and not isinstance(key(d), tuple(excludes))  # type: ignore
    ]
