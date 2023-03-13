from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Type

import numpy as np
import pandas as pd
from pydantic import validate_arguments, validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import AnyConstraint, NChooseKConstraint
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import AnyFeature, ContinuousInput
from bofire.data_models.strategies.validation import (
    validate_constraints,
    validate_features,
    validate_input_feature_count,
)

# TODO: should sampler really get the domain? maybe rather an argument of methods?


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

    type: str
    domain: Domain

    _validate_constraints = validator("domain", allow_reuse=True)(validate_constraints)
    _validate_features = validator("domain", allow_reuse=True)(validate_features)
    _validate_input_feature_count = validator("domain", allow_reuse=True)(
        validate_input_feature_count
    )

    @validate_arguments
    def ask(self, n: int, return_all: bool = True) -> pd.DataFrame:
        """Generates the samples. In the case that `NChooseK` constraints are
        present, per combination `n` samples are generated.

        Args:
            n (Tnum_samples): number of samples to generate.
            return_all (bool, optional): If true all `NchooseK` samples
                are generated, else only `n` samples in total. Defaults to True.

        Returns:
            Dataframe with samples.
        """
        # handle here NChooseK
        if len(self.domain.cnstrs.get(NChooseKConstraint)) > 0:
            _, unused = self.domain.get_nchoosek_combinations()
            samples = []
            for u in unused:
                # create new domain without the nchoosekconstraints
                domain = deepcopy(self.domain)
                domain.constraints = domain.cnstrs.get(excludes=NChooseKConstraint)
                # fix the unused features
                for key in u:
                    feat = domain.inputs.get_by_key(key=key)
                    assert isinstance(feat, ContinuousInput)
                    feat.lower_bound = 0.0
                    feat.upper_bound = 0.0
                # setup then sampler for this situation
                # todo also pass the other args here
                sampler = self.__class__(
                    domain=domain, **self.get_portable_attributes()
                )
                samples.append(sampler.ask(n=n))
            samples = pd.concat(samples, axis=0, ignore_index=True)
            if return_all:
                return samples
            return samples.sample(n=n, replace=False, ignore_index=True)

        samples = self._sample(n)
        if len(samples) != n:
            raise ValueError(f"expected {n} samples, got {len(samples)}.")
        return self.domain.validate_candidates(samples, only_inputs=True)

    @abstractmethod
    def get_portable_attributes(self) -> Dict[str, Any]:
        """Method to return the attributes to port to a new sampler instance when
        sampling from domains with NchooseK constraints.

        Returns:
            Dict[str, Any]: _description_
        """
        pass

    @abstractmethod
    def _sample(self, n: int) -> pd.DataFrame:
        """Abstract method that has to be overwritten in the actual sampler to generate the samples

        Args:
            n (Tnum_samples): number of samples to generate

        Returns:
            pd.DataFrame: Dataframe with samples.
        """
        pass

    @classmethod
    @abstractmethod
    def is_constraint_implemented(cls, my_type: Type[AnyConstraint]) -> bool:
        """Abstract method to check if a specific constraint type is implemented for the sampler

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def is_feature_implemented(cls, my_type: Type[AnyFeature]) -> bool:
        """Abstract method to check if a specific feature type is implemented for the sampler

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        pass
