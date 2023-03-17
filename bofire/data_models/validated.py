import json
from typing import Any, Dict, Optional

import pandas as pd
from pydantic.fields import ModelField


class ValidatedDataFrame(pd.DataFrame):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        if isinstance(val, str):
            val = json.loads(val)
        if isinstance(val, dict):
            val = pd.DataFrame.from_dict(val)
        if isinstance(val, pd.DataFrame):
            val = cls(val)
        if isinstance(val, cls):
            return val
        raise TypeError("expected {cls.__name__}, pd.Dataframe, dict, or str")

    @classmethod
    def __modify_schema__(
        cls, field_schema: Dict[str, Any], field: Optional[ModelField]
    ):
        if field:
            field_schema["type"] = "object"
            field_schema["additionalProperties"] = {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "integer"},
                    ]
                },
            }

    def __eq__(self, other):
        if isinstance(other, pd.DataFrame):
            res = super().__eq__(other)
            while isinstance(res, (pd.DataFrame, pd.Series)):
                res = res.all()
            return res
        raise TypeError(f"cannot compare {self.__class__} to {other.__class__}")


class ValidatedSeries(pd.Series):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        if isinstance(val, str):
            val = json.loads(val)
        if isinstance(val, list):
            val = pd.Series(data=val)
        if isinstance(val, pd.Series):
            val = cls(val)
        if isinstance(val, cls):
            return val
        raise TypeError(f"expected {cls.__name__}, pd.Series, list, or str")

    @classmethod
    def __modify_schema__(
        cls, field_schema: Dict[str, Any], field: Optional[ModelField]
    ):
        if field:
            field_schema["type"] = "array"
            field_schema["items"] = {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "integer"},
                ]
            }

    def __eq__(self, other):
        if isinstance(other, pd.Series):
            res = super().__eq__(other)
            while isinstance(res, pd.Series):
                res = res.all()
            return res
        raise TypeError(f"cannot compare {self.__class__} to {other.__class__}")
