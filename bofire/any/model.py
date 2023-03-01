from typing import Union

from bofire.models.gps import gps
from bofire.models.random_forest import RandomForest

AnyModel = Union[
    gps.SingleTaskGPModel,
    gps.MixedSingleTaskGPModel,
    RandomForest,
]
