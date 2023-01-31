from typing import Union

from bofire.models.torch_models import MixedSingleTaskGPModel, SingleTaskGPModel

AnyModel = Union[
    SingleTaskGPModel,
    MixedSingleTaskGPModel,
]
