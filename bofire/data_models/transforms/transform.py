from typing import Any

from bofire.data_models.base import BaseModel


class Transform(BaseModel):
    """Changes the data a step of a stepwise strategy works with.

    The experiments and pending candidates are transformed before the step's strategy
    sees them, and the candidates it proposes are transformed back.
    """

    type: Any
