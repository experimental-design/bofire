import inspect
from enum import Enum
from typing import List, Tuple, Type

import pandas as pd
from everest.domain.constraints import (Constraint, LinearConstraint,
                                        LinearEqualityConstraint,
                                        LinearInequalityConstraint)
from everest.strategies.strategy import Strategy


class DummyStrategy(Strategy):
    def _init_domain(
        self,
    ) -> None:
        pass

    def _tell(
        self,
    ) -> None:
        pass

    def _ask(
        self,
        candidate_count: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        raise NotImplementedError(
            f"{inspect.stack()[0][3]} not implemented for {self._class_}"
        )

    def has_sufficient_experiments(
        self,
    ) -> bool:
        return len(self.experiments) >= 3

    @classmethod
    def is_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearConstraint,
            LinearEqualityConstraint,
            LinearInequalityConstraint,
        ]
