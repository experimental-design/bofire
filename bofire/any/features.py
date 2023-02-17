from typing import Union

from bofire.domain import features

AnyFeatures = Union[
    features.InputFeatures,
    features.OutputFeatures,
]
