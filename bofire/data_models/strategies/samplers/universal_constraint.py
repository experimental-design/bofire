from typing import Literal, Type

from bofire.data_models.constraints.api import (
    LinearInequalityConstraint,
    LinearEqualityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
    NonlinearEqualityConstraint,
    InterpointEqualityConstraint,
)
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


class UniversalConstraintSampler(SamplerStrategy):
    """Sampler that generates samples by optimization in IPOPT.

    Attributes:
        domain (Domain): Domain defining the constrained input space
        sampling_fraction (float, optional): Fraction of sampled points to total points generated in 
            the sampling process. Defaults to 0.3.
        ipopt_options (dict, optional): Dictionary containing options for the IPOPT solver. Defaults to {"maxiter":200, "disp"=0}.
    """

    type: Literal["UniversalConstraintSampler"] = "UniversalConstraintSampler"
    sampling_fraction: float = 0.3
    ipopt_options: dict = {"maxiter": 200, "disp": 0}

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NonlinearInequalityConstraint,
            NonlinearEqualityConstraint,
            NChooseKConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousInput,
            ContinuousOutput,
        ]
