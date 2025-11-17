from typing import Any, List, Optional

from bofire.data_models.base import BaseModel


class Kernel(BaseModel):
    type: Any


class AggregationKernel(Kernel):
    pass


class FeatureSpecificKernel(Kernel):
    features: Optional[List[str]] = None
