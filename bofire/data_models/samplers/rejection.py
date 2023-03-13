from typing import Any, Dict, Literal, Type

import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import (
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
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
from bofire.data_models.samplers.sampler import Sampler


class RejectionSampler(Sampler):
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

    def _sample(self, n: int) -> pd.DataFrame:
        if len(self.domain.constraints) == 0:
            return self.domain.inputs.sample(n, self.sampling_method)
        n_iters = 0
        n_found = 0
        valid_samples = []
        while n_found < n:
            if n_iters > self.max_iters:
                raise ValueError("Maximum iterations exceeded in rejection sampling.")
            samples = self.domain.inputs.sample(
                self.num_base_samples, method=self.sampling_method
            )
            valid = self.domain.cnstrs.is_fulfilled(samples)
            n_found += np.sum(valid)
            valid_samples.append(samples[valid])
            n_iters += 1
        return pd.concat(valid_samples, ignore_index=True).iloc[:n]

    def get_portable_attributes(self) -> Dict[str, Any]:
        return {
            "sampling_method": self.sampling_method,
            "num_base_samples": self.num_base_samples,
            "max_iters": self.max_iters,
        }

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
        ]
