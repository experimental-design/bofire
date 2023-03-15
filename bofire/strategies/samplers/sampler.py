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
    def ask(self, n: int, return_all: bool = True) -> pd.DataFrame:
        """Generates the samples. In the case that `NChooseK` constraints are
        present, per combination `n` samples are generated.

        Args:
            n (Tnum_asks): number of samples to generate.
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
                sampler: SamplerStrategy = self.duplicate(domain=domain)
                samples.append(sampler.ask(n=n))
            samples = pd.concat(samples, axis=0, ignore_index=True)
            if return_all:
                return samples
            return samples.sample(n=n, replace=False, ignore_index=True)

        samples = self._ask(n)
        if len(samples) != n:
            raise ValueError(f"expected {n} samples, got {len(samples)}.")
        return self.domain.validate_candidates(samples, only_inputs=True)

    def has_sufficient_experiments(self) -> bool:
        return True
