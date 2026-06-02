from typing import Any

from bofire.data_models.base import BaseModel


class Prior(BaseModel):
    """Abstract Prior class."""

    type: Any
