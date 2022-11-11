import inspect
from typing import List, Tuple, Type

import pandas as pd

from bofire.domain.constraints import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.features import ContinuousInput, ContinuousOutput, Feature
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective, Objective
from bofire.strategies.strategy import Strategy


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
            f"{inspect.stack()[0][3]} not implemented for {self.__class__.__name__}"
        )

    def has_sufficient_experiments(
        self,
    ) -> bool:
        return len(self.experiments) >= 3

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearEqualityConstraint,
            LinearInequalityConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [ContinuousInput, ContinuousOutput]

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return my_type in [
            MaximizeObjective,
            MinimizeObjective,
        ]
