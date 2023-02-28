from typing import Union

from bofire.models.gps import gps

AnyModel = Union[
    gps.SingleTaskGPModel,
    gps.MixedSingleTaskGPModel,
]

# # TODO: add RandomForest here
# AnyBotorchModel = Union[
#     gps.SingleTaskGPModel,
#     gps.MixedSingleTaskGPModel,
# ]
