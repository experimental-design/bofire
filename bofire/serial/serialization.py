import json
from typing import Dict

from pydantic import BaseModel


class Serialization:
    """Create serialized data from model instances."""

    @staticmethod
    def dict(data: BaseModel) -> Dict:
        """Return `data.dict()`."""

        return data.dict()

    @staticmethod
    def json(data: BaseModel) -> str:
        """Return `data.json()`."""

        return data.json()

    @staticmethod
    def json_dict(data: BaseModel) -> Dict:
        """Return `json.loads(data.dict())`."""

        return json.loads(data.json())
