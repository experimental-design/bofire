from typing import Any

from bofire.data_models.base import BaseModel


class Transform(BaseModel):
    """Data transformation used in a step of a stepwise strategy.

    Transforms may be applied to the experiments and pending candidates before they
    are passed to the step's strategy, or to the candidates the strategy proposes.
    """

    type: Any
