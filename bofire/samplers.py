import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Literal, Type

import numpy as np
import pandas as pd
import torch
from botorch.utils.sampling import get_polytope_samples
from pydantic import validate_arguments, validator

from bofire.domain.constraint import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
)
from bofire.domain.domain import Domain
from bofire.domain.feature import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
    Tnum_samples,
)
from bofire.domain.util import PydanticBaseModel
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


class Sampler(PydanticBaseModel):
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
    def ask(self, n: Tnum_samples, return_all: bool = True) -> pd.DataFrame:
        """Generates the samples. In the case that `NChooseK` constraints are
        present, per combination `n` samples are generated.

        Args:
            n (Tnum_samples): number of samples to generate.
            return_all (bool, optional): If true all `NchooseK` samples
                are generated, else only `n` samples in total. Defaults to True.

        Returns:
            pd.DataFrame: Dataframe with samples.
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
        fallback_sampling_method: SamplingMethodEnum, optional): Method to use for sampling when no
            constraints are present. Defaults to UNIFORM.
    """

    type: Literal["PolytopeSampler"] = "PolytopeSampler"
    fallback_sampling_method: SamplingMethodEnum = SamplingMethodEnum.UNIFORM

    def _sample(self, n: Tnum_samples) -> pd.DataFrame:
        if len(self.domain.constraints) == 0:
            return self.domain.inputs.sample(n, self.fallback_sampling_method)

        # check if we have pseudo fixed features in the linear equality constraints
        # a pseude fixed is a linear euquality constraint with only one feature included
        # this can happen when fixing features when sampling with NChooseK constraints
        eqs = get_linear_constraints(
            domain=self.domain,
            constraint=LinearEqualityConstraint,  # type: ignore
            unit_scaled=False,
        )
        cleaned_eqs = []
        pseudo_fixed = {}
        for eq in eqs:
            if (
                len(eq[0]) == 1
            ):  # only one coefficient, so this is a pseudo fixed feature
                pseudo_fixed[
                    self.domain.inputs.get_keys(ContinuousInput)[eq[0][0]]
                ] = float(eq[2] / eq[1][0])
            else:
                cleaned_eqs.append(eq)

        # we have to map the indices in case of fixed features
        # as we remove all fixed feature for the sampler, we have to adjust the
        # indices in the constraints, here we get the mapper to map original
        # to adjusted indices
        feature_map = {}
        counter = 0
        for i, feat in enumerate(self.domain.get_features(ContinuousInput)):
            if (not feat.is_fixed()) and (feat.key not in pseudo_fixed.keys()):  # type: ignore
                feature_map[i] = counter
                counter += 1

        # get the bounds
        lower = [
            feat.lower_bound  # type: ignore
            for feat in self.domain.get_features(ContinuousInput)
            if not feat.is_fixed() and feat.key not in pseudo_fixed.keys()  # type: ignore
        ]
        if len(lower) == 0:
            warnings.warn(
                "Nothing to sample, all is fixed. Just the fixed set is returned.",
                UserWarning,
            )
            samples = pd.DataFrame(
                data=np.nan, index=range(n), columns=self.domain.inputs.get_keys()
            )
        else:
            upper = [
                feat.upper_bound  # type: ignore
                for feat in self.domain.get_features(ContinuousInput)
                if not feat.is_fixed() and feat.key not in pseudo_fixed.keys()  # type: ignore
            ]
            bounds = torch.tensor([lower, upper]).to(**tkwargs)
            assert bounds.shape[-1] == len(feature_map) == counter

            # get the inequality constraints and map features back
            # we also check that only features present in the mapper
            # are present in the constraints
            ineqs = get_linear_constraints(
                domain=self.domain,
                constraint=LinearInequalityConstraint,  # type: ignore
                unit_scaled=False,
            )
            for ineq in ineqs:
                for key, value in feature_map.items():
                    if key != value:
                        ineq[0][ineq[0] == key] = value
                assert (
                    ineq[0].max() <= counter
                ), "Something went wrong when transforming the linear constraints. Revisit the problem."

            # map the indice of the equality constraints
            for eq in cleaned_eqs:
                for key, value in feature_map.items():
                    if key != value:
                        eq[0][eq[0] == key] = value
                assert (
                    eq[0].max() <= counter
                ), "Something went wrong when transforming the linear constraints. Revisit the problem."

            # now use the hit and run sampler
            candidates = get_polytope_samples(
                n=n,
                bounds=bounds.to(**tkwargs),
                inequality_constraints=ineqs if len(ineqs) > 0 else None,
                equality_constraints=cleaned_eqs if len(cleaned_eqs) > 0 else None,
                n_burnin=1000,
                # thinning=200
            )

            # check that the random generated candidates are not always the same
            if (candidates.unique(dim=0).shape[0] != n) and (n > 1):
                raise ValueError("Generated candidates are not unique!")

            free_continuals = [
                feat.key
                for feat in self.domain.get_features(ContinuousInput)
                if not feat.is_fixed() and feat.key not in pseudo_fixed.keys()  # type: ignore
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

        # setup the pseudo fixed ones
        for key, value in pseudo_fixed.items():
            samples[key] = value

        return samples

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

    def get_portable_attributes(self) -> Dict[str, Any]:
        return {"fallback_sampling_method": self.fallback_sampling_method}


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
