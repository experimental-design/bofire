from typing import List, Optional

from bofire.data_models.base import BaseModel


class Kernel(BaseModel):
    type: str


class AggregationKernel(Kernel):
    pass


class FeatureSpecificKernel(Kernel):
    features: Optional[List[str]] = None
