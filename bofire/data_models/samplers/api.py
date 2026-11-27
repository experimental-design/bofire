from bofire.data_models.samplers.sampler import (
    LatinHypercubeSampler,
    Sampler,
    SobolSampler,
    UniformSampler,
)
from bofire.data_models.unions import tagged_union


_SAMPLER_TYPES = [
    UniformSampler,
    SobolSampler,
    LatinHypercubeSampler,
]

AnySampler = tagged_union(*_SAMPLER_TYPES)
