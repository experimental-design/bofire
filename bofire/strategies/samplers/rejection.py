import numpy as np
import pandas as pd

from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import RejectionSampler as DataModel
from bofire.strategies.samplers.sampler import SamplerStrategy


class RejectionSampler(SamplerStrategy):
    """Sampler that generates samples via rejection sampling.

    Attributes:
        domain (Domain): Domain defining the constrained input space
        sampling_method (SamplingMethodEnum, optional): Method to generate the base samples. Defaults to UNIFORM.
        num_base_samples (int, optional): Number of base samples to sample in each iteration. Defaults to 1000.
        max_iters (int, optinal): Number of iterations. Defaults to 1000.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.sampling_method = data_model.sampling_method
        self.num_base_samples = data_model.num_base_samples
        self.max_iters = data_model.max_iters

    def _ask(self, n: int) -> pd.DataFrame:
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
            valid = self.domain.constraints.is_fulfilled(samples)
            n_found += np.sum(valid)
            valid_samples.append(samples[valid])
            n_iters += 1
        return pd.concat(valid_samples, ignore_index=True).iloc[:n]

    def duplicate(self, domain: Domain) -> SamplerStrategy:
        data_model = DataModel(
            domain=domain,
            sampling_method=self.sampling_method,
            num_base_samples=self.num_base_samples,
            max_iters=self.max_iters,
        )
        return self.__class__(data_model=data_model)
