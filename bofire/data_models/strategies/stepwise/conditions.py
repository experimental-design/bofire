from typing import List, Literal, Union

from pydantic import Field, validator
from typing_extensions import Annotated

from bofire.data_models.base import BaseModel


class Condition(BaseModel):
    type: str


class SingleCondition(BaseModel):
    type: str


class NumberOfExperimentsCondition(SingleCondition):
    type: Literal["NumberOfExperimentsCondition"] = "NumberOfExperimentsCondition"
    n_experiments: Annotated[int, Field(ge=1)]


class AlwaysTrueCondition(SingleCondition):
    type: Literal["AlwaysTrueCondition"] = "AlwaysTrueCondition"


class CombiCondition(Condition):
    type: Literal["CombiCondition"] = "CombiCondition"
    conditions: Annotated[
        List[
            Union[NumberOfExperimentsCondition, "CombiCondition", AlwaysTrueCondition]
        ],
        Field(min_items=2),
    ]
    n_required_conditions: Annotated[int, Field(ge=0)]

    @validator("n_required_conditions")
    def validate_n_required_conditions(cls, v, values):
        if v > len(values["conditions"]):
            raise ValueError(
                "Number of required conditions larger than number of conditions."
            )
        return v
