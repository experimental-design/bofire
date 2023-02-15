from typing import Union

from bofire.models.gps.gps import MixedSingleTaskGPModel, SingleTaskGPModel

AnyModel = Union[
    SingleTaskGPModel,
    MixedSingleTaskGPModel,
]
