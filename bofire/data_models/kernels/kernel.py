from typing import Any, Optional

from bofire.data_models.base import BaseModel
from bofire.data_models.types import NonRestrictedFeatureKeys


class Kernel(BaseModel):
    type: Any


class AggregationKernel(Kernel):
    pass


class FeatureSpecificKernel(Kernel):
    features: Optional[NonRestrictedFeatureKeys] = None
