import collections.abc as collections
from typing import Any, Callable, List, Sequence, Type, Union, get_args, get_origin

import pandas as pd
from pydantic import BaseModel as _BaseModel
from pydantic import validator


def isinstance_or_union(obj, of):
    if get_origin(of) is Union:
        of = get_args(of)
    return isinstance(obj, of)


def name2key(name):
    # this has been commented out due to some
    # problems with the backend, in the future
    # the backend needs to have a verifier for
    # to validate strings
    # key = re.sub(r'[^a-zA-Z0-9 _/]+', '', name)
    # key = key.strip()
    # key = key.replace(" ", "_")
    # if not len(key) > 0:
    #    raise ValueError("key cannot be empty")
    # return key
    return name


# config details: https://pydantic-docs.helpmanual.io/usage/model_config/
class BaseModel(_BaseModel):
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        copy_on_model_validation = "none"


class KeyModel(BaseModel):
    key: str

    @validator("key")
    def validate_key(cls, v):
        return name2key(v)


def is_numeric(s: pd.Series):
    return pd.to_numeric(s, errors="coerce").notnull().all()


def is_categorical(s: pd.Series, categories: List[str]):
    return sum(s.isin(categories)) == len(s)


def filter_by_attribute(
    data: Sequence,
    attribute_getter: Callable[[Type], Any],
    includes: Union[Type, Sequence[Type]] = None,
    excludes: Union[Type, Sequence[Type]] = None,
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
    includes: Union[Type, Sequence[Type]] = None,
    excludes: Union[Type, Sequence[Type]] = None,
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

    if len([x for x in includes if x in excludes]) > 0:
        raise ValueError("includes and excludes overlap")

    if exact:
        return [
            d for d in data if type(key(d)) in includes and type(key(d)) not in excludes
        ]
    return [
        d
        for d in data
        if isinstance(key(d), tuple(includes))
        and not isinstance(key(d), tuple(excludes))
    ]
