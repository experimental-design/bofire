from copy import deepcopy

import numpy as np
import pandas as pd
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.domain.api import Constraints, Domain
from bofire.strategies.polytope import PolytopeSampler
from bofire.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    """Strategy for randomly selecting new candidates.

    Provides a baseline strategy for benchmarks or for generating initial candidates.
    Uses PolytopeSampler or RejectionSampler, depending on the constraints.
    """

    def __init__(
        self,
        data_model: data_models.RandomStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.num_base_samples = data_model.num_base_samples
        self.max_iters = data_model.max_iters
        self.fallback_sampling_method = data_model.fallback_sampling_method
        self.n_burnin = data_model.n_burnin
        self.n_thinning = data_model.n_thinning

    def _get_feasible_domain_for_polytope_sampler(self) -> Domain:
        domain = deepcopy(self.domain)
        domain.constraints = Constraints(
            constraints=[
                c
                for c in domain.constraints
                if data_models.PolytopeSampler.is_constraint_implemented(type(c))
            ]
        )
        return domain

    def has_sufficient_experiments(self) -> bool:
        return True

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        domain = self._get_feasible_domain_for_polytope_sampler()
        sampler = PolytopeSampler(
            data_model=data_models.PolytopeSampler(
                domain=domain,
                seed=self._get_seed(),
                fallback_sampling_method=self.fallback_sampling_method,
                n_burnin=self.n_burnin,
                n_thinning=self.n_thinning,
            )
        )
        if domain == self.domain:
            return sampler.ask(candidate_count)
        # perform the rejection sampling
        num_base_samples = self.num_base_samples or candidate_count
        n_iters = 0
        n_found = 0
        valid_samples = []
        while n_found < candidate_count:
            if n_iters > self.max_iters:
                raise ValueError("Maximum iterations exceeded in rejection sampling.")
            samples = sampler.ask(
                candidate_count=num_base_samples,
            )
            valid = self.domain.constraints.is_fulfilled(samples)
            n_found += np.sum(valid)
            valid_samples.append(samples[valid])
            n_iters += 1
        return pd.concat(valid_samples, ignore_index=True).iloc[:candidate_count]
