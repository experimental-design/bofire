from typing import Any, Literal

from bofire.data_models.base import BaseModel


class CostAwareUtility(BaseModel):
    """Function mapping the cost of an experiment to its utility."""

    type: Any


class InverseCostWeightedUtility(CostAwareUtility):
    type: Literal["InverseCostWeightedUtility"] = "InverseCostWeightedUtility"
