from typing import Annotated, List, Literal, Union

from pydantic import Field, validator

from bofire.data_models.base import BaseModel


class Condition(BaseModel):
    type: str


class SingleCondition(BaseModel):
    type: str


class RequiredExperimentsCondition(SingleCondition):
    type: Literal["RequiredExperiments"] = "RequiredExperiments"
    n_required_experiments: Annotated[int, Field(ge=0)]


class CombiCondition(Condition):
    type: Literal["CombiCondition"] = "CombiCondition"
    conditions: Annotated[
        List[Union[RequiredExperimentsCondition, "CombiCondition"]], Field(min_items=2)
    ]
    n_required_conditions: Annotated[int, Field(ge=0)]

    @validator("n_required_conditions")
    def validate_n_required_conditions(cls, v, values):
        if v > len(values["conditions"]):
            raise ValueError(
                "Number of required conditions largen than number of conditions."
            )
        return v
