import pandas as pd

from bofire.data_models.strategies.api import SpaceFillingStrategy as DataModel
from bofire.strategies.doe.design import find_local_max_ipopt
from bofire.strategies.enum import OptimalityCriterionEnum
from bofire.strategies.strategy import Strategy


class SpaceFillingStrategy(Strategy):
    """Sampler that generates space filling samples by optimization in IPOPT.

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

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        samples = find_local_max_ipopt(
            domain=self.domain,
            model_type="linear",  # dummy model
            n_experiments=self.num_candidates
            + int(candidate_count / self.sampling_fraction),
            ipopt_options=self.ipopt_options,
            objective=OptimalityCriterionEnum.SPACE_FILLING,
            fixed_experiments=self.candidates,
        )

        samples = samples.iloc[
            self.num_candidates :,
        ]
        samples = samples.sample(
            n=candidate_count,
            replace=False,
            ignore_index=True,
            random_state=self._get_seed(),
        )

        self.domain.validate_experiments(samples)

        return samples

    def has_sufficient_experiments(self) -> bool:
        return True
