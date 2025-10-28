from typing import Literal

from bofire.data_models.constraints.condition import Condition
from bofire.data_models.features.continuous import ContinuousInput


class ConditionalContinuousInput(ContinuousInput):
    """An input that is only active when another feature meets some condition."""

    type: Literal["ConditionalContinuousInput"] = "ConditionalContinuousInput"  # type: ignore
    indicator_feature: str
    indicator_condition: Condition
