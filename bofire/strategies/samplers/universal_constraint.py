import numpy as np
import pandas as pd

from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import UniversalConstraintSampler as DataModel
from bofire.strategies.samplers.sampler import SamplerStrategy
from bofire.strategies.doe.design import find_local_max_ipopt

from bofire.strategies.enum import OptimalityCriterionEnum


class UniversalConstraintSampler(SamplerStrategy):
    """Sampler that generates samples by optimization in IPOPT.

    Attributes:
        domain (Domain): Domain defining the constrained input space
        sampling_fraction (float, optional): Fraction of sampled points to total points generated in 
            the sampling process. Defaults to 0.3.
        ipopt_options (dict, optional): Dictionary containing options for the IPOPT solver. Defaults to {"maxiter":200, "disp"=0}.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        assert data_model.sampling_fraction > 0 and data_model.sampling_fraction <= 1
        self.sampling_fraction = data_model.sampling_fraction
        self.ipopt_options = data_model.ipopt_options

    def _ask(self, n: int) -> pd.DataFrame:
        samples = find_local_max_ipopt(
            domain = self.domain,
            model_type="linear",    # dummy model
            n_experiments=int(n/self.sampling_fraction),
            ipopt_options=self.ipopt_options,
            objective=OptimalityCriterionEnum.SPACE_FILLING,
        )

        samples = samples.iloc[np.random.choice(len(samples), n, replace=False),:]
        samples.index = range(n)

        self.domain.validate_experiments(samples)

        return samples

    def duplicate(self, domain: Domain) -> SamplerStrategy:
        data_model = DataModel(
            domain=domain,
            sampling_fraction=self.sampling_fraction,
            ipopt_options=self.ipopt_options,
        )
        return self.__class__(data_model=data_model)
    
