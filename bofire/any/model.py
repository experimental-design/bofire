from typing import Union

from bofire.models.gps import gps

AnyModel = Union[
    gps.SingleTaskGPModel,
    gps.MixedSingleTaskGPModel,
]
