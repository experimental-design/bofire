from typing import Union

from bofire import samplers

AnySampler = Union[
    samplers.PolytopeSampler,
    samplers.RejectionSampler,
]
