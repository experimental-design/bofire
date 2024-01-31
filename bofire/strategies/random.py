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

    This strategy generates candidate samples using the random strategy. It first checks if the domain
    is compatible with the PolytopeSampler. If so, it uses the PolytopeSampler to generate candidate
    samples. If not, it performs rejection sampling by repeatedly generating candidates with the PolytopeSampler
    until the desired number of valid samples is obtained.

    Args:
        data_model (data_models.RandomStrategy): The data model for the random strategy.
        **kwargs: Additional keyword arguments.
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

    @staticmethod
    def _get_feasible_domain_for_polytope_sampler(domain: Domain) -> Domain:
        """
        Returns a modified domain object with only the constraints that are implemented
        for the PolytopeSampler.

        Args:
            domain (Domain): The original domain object.

        Returns:
            Domain: A modified domain object with only the feasible constraints.
        """
        domain = deepcopy(domain)
        domain.constraints = Constraints(
            constraints=[
                c
                for c in domain.constraints
                if data_models.PolytopeSampler.is_constraint_implemented(type(c))
            ]
        )
        return domain

    def has_sufficient_experiments(self) -> bool:
        """
        Check if there are sufficient experiments for the strategy.

        Returns:
            bool: True if there are sufficient experiments, False otherwise.
        """
        return True

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        """
        Generate candidate samples using the random strategy.

        If the domain is compatible with the PolytopeSampler, it uses the PolytopeSampler to generate
        candidate samples. Otherwise, it performs rejection sampling by repeatedly generating candidate
        samples until the desired number of valid samples is obtained.

        Args:
            candidate_count (PositiveInt): The number of candidate samples to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the generated candidate samples.
        """
        domain = self._get_feasible_domain_for_polytope_sampler(domain=self.domain)
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
