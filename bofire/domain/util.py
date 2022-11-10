from typing import Any, Callable, List, Type, Union

import pandas as pd
from pydantic import BaseModel as _BaseModel
from pydantic import validator


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
    data: List,
    attribute_getter: Callable[[Type], Any],
    includes: Union[Type, List[Type]] = None,
    excludes: Union[Type, List[Type]] = None,
    exact: bool = False,
):
    data_with_attr = []
    for d in data:
        try:
            attribute_getter(d)
            data_with_attr.append(d)
        except AttributeError:
            pass

    filtered = filter_by_class(
        [(i, attribute_getter(d)) for i, d in enumerate(data_with_attr)],
        includes=includes,
        excludes=excludes,
        exact=exact,
        key=lambda idx_desi_pair: idx_desi_pair[1],
    )
    if len(filtered) == 0:
        return filtered
    else:
        out_indices, _ = zip(*filtered)
        return [data_with_attr[i] for i in out_indices]


def filter_by_class(
    data: List,
    includes: Union[Type, List[Type]] = None,
    excludes: Union[Type, List[Type]] = None,
    exact: bool = False,
    key: Callable[[Type], Any] = lambda x: x,
) -> List:
    if includes is None:
        includes = []
    if not isinstance(includes, list):
        includes = [includes]
    if excludes is None:
        excludes = []
    if not isinstance(excludes, list):
        excludes = [excludes]

    if len(includes) == len(excludes) == 0:
        raise ValueError("no filter provided")

    if len(includes) == 0:
        includes = [object]

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
