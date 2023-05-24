import math
from abc import abstractmethod
from copy import deepcopy

import pandas as pd
from pydantic import validate_arguments

from bofire.data_models.constraints.api import NChooseKConstraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.strategies.api import SamplerStrategy as DataModel
from bofire.strategies.strategy import Strategy


class SamplerStrategy(Strategy):
    """Base class for sampling methods in BoFire for sampling from constrained input spaces.

    Attributes
        domain (Domain): Domain defining the constrained input space
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    @abstractmethod
    def duplicate(self, domain: Domain) -> "SamplerStrategy":
        pass

    @validate_arguments
    def ask(
        self,
        candidate_count: int,
        return_all: bool = False,
        raise_validation_error: bool = True,
    ) -> pd.DataFrame:
        """Generates the samples. In the case that `NChooseK` constraints are
        present, per combination `n` samples are generated.

        Args:
            n (Tnum_asks): number of samples to generate.
            return_all (bool, optional): If true all `NchooseK` samples
                are generated, else only `n` samples in total. Defaults to True.
            raise_validation_error (bool, optional): If true an error will be raised if candidates violate constraints,
                otherwise only a warning will be displayed. Defaults to True.

        Returns:
            Dataframe with samples.
        """
        # n = candidate_count
        # handle here NChooseK
        if len(self.domain.constraints.get(NChooseKConstraint)) > 0:
            _, unused = self.domain.get_nchoosek_combinations()

            if return_all:
                sampled_combinations = unused
                num_samples_per_it = candidate_count
            else:
                if candidate_count <= len(unused):
                    sampled_combinations = [
                        unused[i]
                        for i in self.rng.choice(
                            len(unused), size=candidate_count, replace=False
                        )
                    ]
                    num_samples_per_it = 1
                else:
                    sampled_combinations = unused
                    num_samples_per_it = math.ceil(candidate_count / len(unused))

            samples = []
            for u in sampled_combinations:
                # create new domain without the nchoosekconstraints
                domain = deepcopy(self.domain)
                domain.constraints = domain.constraints.get(excludes=NChooseKConstraint)
                # fix the unused features
                for key in u:
                    feat = domain.inputs.get_by_key(key=key)
                    assert isinstance(feat, ContinuousInput)
                    feat.bounds = (0, 0)
                # setup then sampler for this situation
                sampler: SamplerStrategy = self.duplicate(domain=domain)
                samples.append(sampler.ask(num_samples_per_it))
            samples = pd.concat(samples, axis=0, ignore_index=True)
            if return_all:
                return self.domain.validate_candidates(
                    samples,
                    only_inputs=True,
                    raise_validation_error=raise_validation_error,
                )
            return self.domain.validate_candidates(
                samples.sample(n=candidate_count, replace=False, ignore_index=True),
                only_inputs=True,
                raise_validation_error=raise_validation_error,
            )

        samples = self._ask(candidate_count)
        if len(samples) != candidate_count:
            raise ValueError(f"expected {candidate_count} samples, got {len(samples)}.")
        return self.domain.validate_candidates(
            samples, only_inputs=True, raise_validation_error=raise_validation_error
        )

    def has_sufficient_experiments(self) -> bool:
        return True
