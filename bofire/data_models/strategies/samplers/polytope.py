from typing import Annotated, Literal, Type

from pydantic import Field

from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
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
    n_burnin: Annotated[int, Field(ge=1)] = 1000
    n_thinning: Annotated[int, Field(ge=1)] = 32

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            LinearInequalityConstraint,
            LinearEqualityConstraint,
            NChooseKConstraint,
            InterpointEqualityConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousInput,
            ContinuousOutput,
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
            CategoricalMolecularInput,
        ]
