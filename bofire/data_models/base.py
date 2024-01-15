import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    # json_encoders is deprecated.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=False,
        extra="forbid",
        json_encoders={
            pd.DataFrame: lambda x: x.to_dict(orient="list"),
            pd.Series: lambda x: x.to_list(),
        },
    )
