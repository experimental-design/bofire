from typing import Optional

import pandas as pd
from pydantic.error_wrappers import ValidationError
from pydantic.types import PositiveInt

from bofire.data_models.api import AnySampler
from bofire.data_models.samplers.api import PolytopeSampler, RejectionSampler
from bofire.data_models.strategies.api import RandomStrategy as DataModel
from bofire.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    """Strategy for randomly selecting new candidates.

    Provides a baseline strategy for benchmarks or for generating initial candidates.
    Uses PolytopeSampler or RejectionSampler, depending on the constraints.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self._init_sampler()

    sampler: Optional[AnySampler] = None

    def _init_sampler(self) -> None:
        errors = []
        for S in [PolytopeSampler, RejectionSampler]:
            try:
                self.sampler = S(domain=self.domain)
                break
            except ValidationError as err:
                errors.append(err)
        if self.sampler is None:
            raise Exception(errors)

    def has_sufficient_experiments(self) -> bool:
        return True

    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[PositiveInt] = None,
    ) -> pd.DataFrame:
        return candidate_pool.sample(n=candidate_count)

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        return self.sampler.ask(candidate_count)  # type: ignore
