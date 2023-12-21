import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra


class BaseModel(PydanticBaseModel):
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = False
        copy_on_model_validation = "none"
        extra = Extra.forbid

        json_encoders = {
            pd.DataFrame: lambda x: x.to_dict(orient="list"),
            pd.Series: lambda x: x.to_list(),
        }
