from typing import Union

from bofire.data_models.samplers.polytope import PolytopeSampler
from bofire.data_models.samplers.rejection import RejectionSampler
from bofire.data_models.samplers.sampler import Sampler

AbstractSampler = Sampler

AnySampler = Union[RejectionSampler, PolytopeSampler]
