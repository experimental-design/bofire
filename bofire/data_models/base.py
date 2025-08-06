from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )
