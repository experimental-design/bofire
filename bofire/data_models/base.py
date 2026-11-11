from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    """Base of all BoFire data models: unknown fields are rejected, and assignments
    are validated like construction."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )
