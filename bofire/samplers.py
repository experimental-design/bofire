from abc import abstractmethod
from typing import Type

import numpy as np
import pandas as pd
import torch
from botorch.utils.sampling import get_polytope_samples
from pydantic import validate_arguments, validator

from bofire.domain import Domain
from bofire.domain.constraints import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
)
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
    Tnum_samples,
)
from bofire.domain.util import BaseModel
from bofire.strategies.strategy import (
    validate_constraints,
    validate_features,
    validate_input_feature_count,
)
from bofire.utils.enum import SamplingMethodEnum
from bofire.utils.torch_tools import get_linear_constraints, tkwargs


def apply_nchoosek(samples: pd.DataFrame, constraint: NChooseKConstraint):
    """Apply an NChooseK constraint in place.

    Args:
        samples (pd.DataFrame): Dataframe with samples
        constraint (NChooseKConstraint): NChooseK constraint which should be applied.
    """
    n_zeros = len(constraint.features) - constraint.max_count
    for i in samples.index:
        s = np.random.choice(constraint.features, size=n_zeros, replace=False)
        samples.loc[i, s] = 0


class Sampler(BaseModel):
    """Base class for sampling methods in BoFire for sampling from constrained input spaces.

    Attributes
        domain (Domain): Domain defining the constrained input space
    """

    domain: Domain

    _validate_constraints = validator("domain", allow_reuse=True)(validate_constraints)
    _validate_features = validator("domain", allow_reuse=True)(validate_features)
    _validate_input_feature_count = validator("domain", allow_reuse=True)(
        validate_input_feature_count
    )

    @validate_arguments
    def ask(self, n: Tnum_samples) -> pd.DataFrame:
        """Generates the samples.

        Args:
            n (Tnum_samples): number of samples to generate

        Returns:
            pd.DataFrame: Dataframe with samples.
        """
        samples = self._sample(n)
        if len(samples) != n:
            raise ValueError(f"expected {n} samples, got {len(samples)}.")
        return self.domain.validate_candidates(samples, only_inputs=True)

    @abstractmethod
    def _sample(self, n: Tnum_samples) -> pd.DataFrame:
        """Abstract method that has to be overwritten in the actual sampler to generate the samples

        Args:
            n (Tnum_samples): number of samples to generate

        Returns:
            pd.DataFrame: Dataframe with samples.
        """
        pass

    @classmethod
    @abstractmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Abstract method to check if a specific constraint type is implemented for the sampler

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Abstract method to check if a specific feature type is implemented for the sampler

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        pass


class PolytopeSampler(Sampler):
    """Sampler that generates samples from a Polytope defined by linear equality and ineqality constraints.

    Attributes:
        domain (Domain): Domain defining the constrained input space
    """

    def _sample(self, n: Tnum_samples) -> pd.DataFrame:
        if len(self.domain.constraints) == 0:
            return self.domain.inputs.sample(n, SamplingMethodEnum.UNIFORM)

        # get the bounds
        lower = [
            feat.lower_bound  # type: ignore
            for feat in self.domain.get_features(ContinuousInput)
            if not feat.is_fixed()  # type: ignore
        ]
        upper = [
            feat.upper_bound  # type: ignore
            for feat in self.domain.get_features(ContinuousInput)
            if not feat.is_fixed()  # type: ignore
        ]
        bounds = torch.tensor([lower, upper]).to(**tkwargs)

        # now use the hit and run sampler
        candidates = get_polytope_samples(
            n=n,
            bounds=bounds.to(**tkwargs),
            inequality_constraints=get_linear_constraints(
                domain=self.domain,
                constraint=LinearInequalityConstraint,  # type: ignore
                unit_scaled=False,
            ),
            equality_constraints=get_linear_constraints(
                domain=self.domain,
                constraint=LinearEqualityConstraint,  # type: ignore
                unit_scaled=False,
            ),
            n_burnin=1000,
            # thinning=200
        )

        # check that the random generated candidates are not always the same
        if (candidates.unique(dim=0).shape[0] != n) and (n > 1):
            raise ValueError("Generated candidates are not unique!")

        free_continuals = [
            feat.key
            for feat in self.domain.get_features(ContinuousInput)
            if not feat.is_fixed()  # type: ignore
        ]

        # setup the output
        samples = pd.DataFrame(
            data=candidates.detach().numpy().reshape(n, len(free_continuals)),
            index=range(n),
            columns=free_continuals,
        )

        # setup the categoricals and discrete ones as uniform sampled vals
        for feat in self.domain.get_features([CategoricalInput, DiscreteInput]):
            samples[feat.key] = feat.sample(n)  # type: ignore

        # setup the fixed continuous ones
        for feat in self.domain.inputs.get_fixed():
            samples[feat.key] = feat.fixed_value()[0]  # type: ignore

        return samples

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [LinearInequalityConstraint, LinearEqualityConstraint]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousInput,
            ContinuousOutput,
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
        ]


class RejectionSampler(Sampler):
    """Sampler that generates samples via rejection sampling.

    Attributes:
        domain (Domain): Domain defining the constrained input space
        sampling_method (SamplingMethodEnum, optional): Method to generate the base samples. Defaults to UNIFORM.
        num_base_samples (int, optional): Number of base samples to sample in each iteration. Defaults to 1000.
        max_iters (int, optinal): Number of iterations. Defaults to 1000.
    """

    sampling_method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM
    num_base_samples: Tnum_samples = 1000
    max_iters: Tnum_samples = 1000

    def _sample(self, n: Tnum_samples) -> pd.DataFrame:
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
