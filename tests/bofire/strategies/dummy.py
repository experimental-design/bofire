import inspect
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from pydantic.types import NonNegativeInt

from bofire.domain.constraints import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.features import ContinuousInput, ContinuousOutput, Feature
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective, Objective
from bofire.strategies.strategy import PredictiveStrategy, Strategy
from bofire.utils.transformer import Transformer


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

    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[NonNegativeInt] = None,
    ) -> pd.DataFrame:
        candidates = candidate_pool.sample(candidate_count, replace=False)
        for feat in self.domain.get_outputs_by_objective(Objective):
            candidates[f"{feat.key}_pred"] = np.nan
            candidates[f"{feat.key}_sd"] = np.nan
            candidates[f"{feat.key}_des"] = np.nan
        return candidates

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


class DummyPredictiveStrategy(PredictiveStrategy):
    def _init_domain(
        self,
    ) -> None:
        # self._is_fitted = False
        self.transformer = Transformer(self)

    def _tell(
        self,
    ) -> None:
        pass

    def _fit(self, transformed: pd.DataFrame):
        pass

    def _predict(self, experiments: pd.DataFrame):
        return (
            np.ones([len(experiments), len(self.domain.output_features)]) * 4,
            np.ones([len(experiments), len(self.domain.output_features)]) * 5,
        )

    def _ask(
        self,
        candidate_count: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        raise NotImplementedError(
            f"{inspect.stack()[0][3]} not implemented for {self.__class__.__name__}"
        )

    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[NonNegativeInt] = None,
    ) -> pd.DataFrame:
        candidates = candidate_pool.sample(candidate_count, replace=False)
        for feat in self.domain.get_outputs_by_objective(Objective):
            candidates[f"{feat.key}_pred"] = np.nan
            candidates[f"{feat.key}_sd"] = np.nan
            candidates[f"{feat.key}_des"] = np.nan
        return candidates

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
