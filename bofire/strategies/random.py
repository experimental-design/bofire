import pandas as pd
from pydantic.error_wrappers import ValidationError
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.strategies.samplers.polytope import PolytopeSampler
from bofire.strategies.samplers.rejection import RejectionSampler
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
        self._init_sampler()

    def _init_sampler(self) -> None:
        errors = []
        for dm, functional in [
            (data_models.PolytopeSampler, PolytopeSampler),
            (data_models.RejectionSampler, RejectionSampler),
        ]:
            try:
                data_model = dm(domain=self.domain)
                self.sampler = functional(data_model=data_model)
                break
            except ValidationError as err:
                errors.append(err)
        if self.sampler is None:
            raise Exception(errors)

    def has_sufficient_experiments(self) -> bool:
        return True

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        return self.sampler.ask(candidate_count)  # type: ignore
