from typing import Literal, Type

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
)
from bofire.data_models.strategies.samplers.sampler import SamplerStrategy


class PolytopeSampler(SamplerStrategy):
    """Sampler that generates samples from a Polytope defined by linear equality and ineqality constraints.

    Attributes:
        domain (Domain): Domain defining the constrained input space
        fallback_sampling_method: SamplingMethodEnum, optional): Method to use for sampling when no
            constraints are present. Defaults to UNIFORM.
    """

    type: Literal["PolytopeSampler"] = "PolytopeSampler"
    fallback_sampling_method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            LinearInequalityConstraint,
            LinearEqualityConstraint,
            NChooseKConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousInput,
            ContinuousOutput,
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
        ]
