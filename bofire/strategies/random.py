from typing import Optional, Type

import pandas as pd
from pydantic.error_wrappers import ValidationError
from pydantic.types import PositiveInt

from bofire.domain.constraints import (
    Constraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
)
from bofire.domain.features import Feature
from bofire.domain.objectives import Objective
from bofire.samplers import PolytopeSampler, RejectionSampler, Sampler
from bofire.strategies.strategy import Strategy


class RandomStrategy(Strategy):
    """Strategy for randomly selecting new candidates.

    Provides a baseline strategy for benchmarks or for generating initial candidates.
    Uses PolytopeSampler or RejectionSampler, depending on the constraints.
    """

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

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        if my_type in [NChooseKConstraint, NonlinearEqualityConstraint]:
            return False
        return True

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
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
        return self.sampler.ask(candidate_count)  # type: ignore
