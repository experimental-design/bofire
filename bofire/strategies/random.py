from typing import Optional

import pandas as pd
from pydantic.types import PositiveInt

from bofire.samplers import RejectionSampler, Sampler
from bofire.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    sampler: Optional[Sampler] = None

    def _init_domain(self) -> None:
        self.sampler = RejectionSampler(domain=self.domain)

    def is_constraint_implemented(self) -> bool:
        return True

    def is_feature_implemented(self) -> bool:
        return True

    def is_objective_implemented(self) -> bool:
        return True

    def has_sufficient_experiments(self) -> bool:
        return True

    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[PositiveInt] = None,
    ) -> pd.DataFrame:
        return candidate_pool.sample(n=candidate_count)

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        return self.sampler.ask(candidate_count)
