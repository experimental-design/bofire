from typing import List, Literal, Optional

from bofire.data_models.base import BaseModel


class Kernel(BaseModel):
    type: str


class AggregationKernel(Kernel):
    pass


class ConcreteKernel(Kernel):
    features: Optional[List[str]] = None
