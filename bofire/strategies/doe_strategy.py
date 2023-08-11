import pandas as pd
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.strategies.doe.design import find_local_max_ipopt
from bofire.strategies.strategy import Strategy


class DoEStrategy(Strategy):
    """Strategy for design of experiments. This strategy is used to generate a set of
    experiments for a given domain.
    The experiments are generated via minimization of the D-optimality criterion.

    """

    def __init__(
        self,
        data_model: data_models.DoEStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self.formula = data_model.formula

    def _ask(self, candidate_count: PositiveInt) -> pd.DataFrame:
        if self.candidates is not None:
            _fixed_experiments_count = len(self.candidates)
            _candidate_count = candidate_count + len(self.candidates)
        else:
            _fixed_experiments_count = 0
            _candidate_count = candidate_count

        design = find_local_max_ipopt(
            self.domain,
            self.formula,
            n_experiments=_candidate_count,
            fixed_experiments=self.candidates,
        )
        return design.iloc[_fixed_experiments_count:, :]  # type: ignore

    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are available.

        Returns:
            bool: True if number of passed experiments is sufficient, False otherwise
        """
        return True
