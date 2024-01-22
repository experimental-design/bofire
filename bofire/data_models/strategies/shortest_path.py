import math
from typing import Annotated, Dict, Literal, Type, Union

import pandas as pd
from pydantic import Field, field_validator, model_validator

from bofire.data_models.constraints.api import (
    ConstraintNotFulfilledError,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import ContinuousInput, Feature
from bofire.data_models.strategies.strategy import Strategy


def has_local_search_region(domain: Domain) -> bool:
    if len(domain.inputs.get(ContinuousInput)) == 0:
        return False
    is_lsr = False
    for feat in domain.inputs.get(ContinuousInput):
        assert isinstance(feat, ContinuousInput)
        if feat.local_relative_bounds != (math.inf, math.inf):
            is_lsr = True
    return is_lsr


class ShortestPathStrategy(Strategy):
    type: Literal["ShortestPathStrategy"] = "ShortestPathStrategy"
    start: Annotated[Dict[str, Union[float, str]], Field(min_length=1)]
    end: Annotated[Dict[str, Union[float, str]], Field(min_length=1)]
    atol: Annotated[float, Field(gt=0)] = 1e-7

    @model_validator(mode="after")
    def validate_start_end(self):
        df_start = pd.DataFrame(pd.Series(self.start)).T
        df_end = pd.DataFrame(pd.Series(self.end)).T
        try:
            self.domain.validate_candidates(df_start, only_inputs=True)
        except (ValueError, ConstraintNotFulfilledError):
            raise ValueError("`start` is not a valid candidate.")
        try:
            self.domain.validate_candidates(df_end, only_inputs=True)
        except (ValueError, ConstraintNotFulfilledError):
            raise ValueError("`end` is not a valid candidate.")

        self.domain.inputs.validate_candidates(df_end)
        # check that start and end are not the same
        if df_start[self.domain.inputs.get_keys()].equals(
            df_end[self.domain.inputs.get_keys()]
        ):
            raise ValueError("`start` is equal to `end`.")
        return self

    @field_validator("domain")
    @classmethod
    def validate_lsr(cls, domain):
        if has_local_search_region(domain=domain) is False:
            raise ValueError("Domain has no local search region.")
        return domain

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [LinearInequalityConstraint, LinearEqualityConstraint]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True
