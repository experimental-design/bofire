from typing import Optional

import pandas as pd

from bofire.data_models.strategies.api import FactorialStrategy as DataModel
from bofire.strategies.strategy import Strategy


class FactorialStrategy(Strategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _ask(self, candidate_count: Optional[int] = None) -> pd.DataFrame:
        if candidate_count is not None:
            raise ValueError(
                "FactorialStrategy will ignore the specified value of candidate_count. "
                "The strategy automatically determines how many candidates to "
                "propose.",
            )
        return pd.DataFrame(
            [
                {e[0]: e[1] for e in combi}
                for combi in self.domain.inputs.get_categorical_combinations()
            ],
        )

    def has_sufficient_experiments(self) -> bool:
        return True
