from typing import Literal, Type

from bofire.data_models.constraints.api import (
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
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


class RejectionSampler(SamplerStrategy):
    """Sampler that generates samples via rejection sampling.

    Attributes:
        domain (Domain): Domain defining the constrained input space
        sampling_method (SamplingMethodEnum, optional): Method to generate the base samples. Defaults to UNIFORM.
        num_base_samples (int, optional): Number of base samples to sample in each iteration. Defaults to 1000.
        max_iters (int, optinal): Number of iterations. Defaults to 1000.
    """

    type: Literal["RejectionSampler"] = "RejectionSampler"
    sampling_method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM
    num_base_samples: int = 1000
    max_iters: int = 1000

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            LinearInequalityConstraint,
            NonlinearInequalityConstraint,
            NChooseKConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousInput,
            ContinuousOutput,
            DiscreteInput,
            CategoricalInput,
            CategoricalDescriptorInput,
            CategoricalMolecularInput,
        ]
