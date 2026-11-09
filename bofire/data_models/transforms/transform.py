from typing import Any

from bofire.data_models.base import BaseModel


class Transform(BaseModel):
    """Data transformation used in a step of a stepwise strategy.

    It transforms the experiments and the pending candidates before the step's
    strategy receives them, and transforms the candidates the strategy proposes back.
    """

    type: Any
