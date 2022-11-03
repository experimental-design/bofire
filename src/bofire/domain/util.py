from typing import List, Type, Union

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


def filter_by_class(
    data: List,
    includes: Union[Type, List[Type]],
    excludes: Union[Type, List[Type]] = None,
    exact: bool = False,
) -> List:
    if not isinstance(includes, list):
        includes = [includes]
    if excludes is None:
        excludes = []
    if not isinstance(excludes, list):
        excludes = [excludes]

    if len(includes) == 0:
        raise ValueError("no filter provided")
    if len([x for x in includes if x in excludes]) > 0:
        raise ValueError("includes and excludes overlap")

    if exact:
        return [d for d in data if type(d) in includes and type(d) not in excludes]
    return [
        d
        for d in data
        if isinstance(d, tuple(includes)) and not isinstance(d, tuple(excludes))
    ]
