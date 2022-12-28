from typing import Optional

import pandas as pd
from pydantic.error_wrappers import ValidationError
from pydantic.types import PositiveInt

from bofire.samplers import PolytopeSampler, RejectionSampler, Sampler
from bofire.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    sampler: Optional[Sampler] = None

    def _init_domain(self) -> None:
        errors = []
        for S in [PolytopeSampler, RejectionSampler]:
            try:
                self.sampler = S(domain=self.domain)
                break
            except ValidationError as err:
                errors.append(err)
        if self.sampler is None:
            raise Exception(errors)

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
