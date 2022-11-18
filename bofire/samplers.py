from abc import abstractmethod
from typing import Type

import numpy as np
import pandas as pd
from pydantic import validate_arguments, validator

from bofire.domain import Domain
from bofire.domain.constraints import Constraint, LinearInequalityConstraint
from bofire.domain.features import (
    ContinuousInput,
    ContinuousOutput,
    Feature,
    SamplingMethodEnum,
    Tnum_samples,
)
from bofire.domain.util import BaseModel
from bofire.strategies.strategy import (
    validate_constraints,
    validate_features,
    validate_input_feature_count,
)


class Sampler(BaseModel):

    domain: Domain

    _validate_constraints = validator("domain", allow_reuse=True)(validate_constraints)
    _validate_features = validator("domain", allow_reuse=True)(validate_features)
    _validate_input_feature_count = validator("domain", allow_reuse=True)(
        validate_input_feature_count
    )

    @validate_arguments
    def __call__(self, n: Tnum_samples) -> pd.DataFrame:
        samples = self._sample(n)
        if len(samples) != n:
            raise ValueError(f"expected {n} samples, got {len(samples)}.")
        return self.domain.validate_candidates(samples, only_inputs=True)

    @abstractmethod
    def _sample(self, n: Tnum_samples) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Abstract method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Abstract method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        pass


class ConstrainedSampler(Sampler):
    def _sample(self, n: Tnum_samples) -> pd.DataFrame:
        return super()._sample(n)


class RejectionSampler(Sampler):

    sampling_method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM
    num_base_samples: int = 1000
    max_iters: int = 1000

    def _sample(self, n: Tnum_samples) -> pd.DataFrame:
        if len(self.domain.constraints) == 0:
            return self.domain.input_features.sample(n, self.sampling_method)
        n_iters = 0
        n_found = 0
        valid_samples = []
        while n_found < n:
            if n_iters > self.max_iters:
                raise Exception("Maximum iterations exceeded in rejection sampling.")
            samples = self.domain.input_features.sample(
                self.num_base_samples, method=self.sampling_method
            )
            valid = self.domain.constraints.is_fulfilled(samples)
            n_found += np.sum(valid)
            valid_samples.append(samples[valid])
            n_iters += 1
        return pd.concat(valid_samples, ignore_index=True).iloc[:n]

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [LinearInequalityConstraint]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [ContinuousInput, ContinuousOutput]
