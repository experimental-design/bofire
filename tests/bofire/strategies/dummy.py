import inspect
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import pandas as pd
from botorch.acquisition.acquisition import AcquisitionFunction
from pydantic.types import NonNegativeInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput, Feature
from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.strategies.api import BotorchStrategy, PredictiveStrategy, Strategy


class DummyStrategyDataModel(data_models.BotorchStrategy):
    type: Literal["DummyStrategyDataModel"] = "DummyStrategyDataModel"  # type: ignore

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
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


class DummyStrategy(Strategy):
    def __init__(
        self,
        data_model: DummyStrategyDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _init_domain(
        self,
    ) -> None:
        pass

    def _tell(
        self,
    ) -> None:
        pass

    def _ask(  # type: ignore
        self,
        candidate_count: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        raise NotImplementedError(
            f"{inspect.stack()[0][3]} not implemented for {self.__class__.__name__}",
        )

    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[NonNegativeInt] = None,
    ) -> pd.DataFrame:
        candidates = candidate_pool.sample(candidate_count, replace=False)
        for feat in self.domain.outputs.get_by_objective(Objective):
            candidates[f"{feat.key}_pred"] = np.nan
            candidates[f"{feat.key}_sd"] = np.nan
            candidates[f"{feat.key}_des"] = np.nan
        return candidates

    def has_sufficient_experiments(
        self,
    ) -> bool:
        return len(self.experiments) >= 3  # type: ignore


class DummyPredictiveStrategyDataModel(data_models.PredictiveStrategy):
    type: Literal["DummyPredictiveStrategyDataModel"] = (  # type: ignore
        "DummyPredictiveStrategyDataModel"
    )

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
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
    def __init__(
        self,
        data_model: DummyPredictiveStrategyDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _init_domain(
        self,
    ) -> None:
        pass

    def _tell(
        self,
    ) -> None:
        pass

    def _fit(self, transformed: pd.DataFrame):  # type: ignore
        pass

    def _predict(self, experiments: pd.DataFrame):  # type: ignore
        return (
            np.ones([len(experiments), len(self.domain.outputs)]) * 4,
            np.ones([len(experiments), len(self.domain.outputs)]) * 5,
        )

    def _ask(  # type: ignore
        self,
        candidate_count: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        raise NotImplementedError(
            f"{inspect.stack()[0][3]} not implemented for {self.__class__.__name__}",
        )

    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[NonNegativeInt] = None,
    ) -> pd.DataFrame:
        candidates = candidate_pool.sample(candidate_count, replace=False)
        for feat in self.domain.outputs.get_by_objective(Objective):
            candidates[f"{feat.key}_pred"] = np.nan
            candidates[f"{feat.key}_sd"] = np.nan
            candidates[f"{feat.key}_des"] = np.nan
        return candidates

    @property
    def input_preprocessing_specs(self):
        return {}

    def has_sufficient_experiments(
        self,
    ) -> bool:
        return len(self.experiments) >= 3  # type: ignore


class DummyBotorchPredictiveStrategy(BotorchStrategy):
    def __init__(
        self,
        data_model: DummyStrategyDataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

    def _init_domain(
        self,
    ) -> None:
        pass

    def _predict(self, experiments: pd.DataFrame):  # type: ignore
        return (
            np.ones([len(experiments), len(self.domain.outputs)]) * 4,
            np.ones([len(experiments), len(self.domain.outputs)]) * 5,
        )

    def _ask(  # type: ignore
        self,
        candidate_count: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        raise NotImplementedError(
            f"{inspect.stack()[0][3]} not implemented for {self.__class__.__name__}",
        )

    def _get_acqfs(self, n: int) -> List[AcquisitionFunction]:
        return []

    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[NonNegativeInt] = None,
    ) -> pd.DataFrame:
        candidates = candidate_pool.sample(candidate_count, replace=False)
        for feat in self.domain.outputs.get_by_objective(Objective):
            candidates[f"{feat.key}_pred"] = np.nan
            candidates[f"{feat.key}_sd"] = np.nan
            candidates[f"{feat.key}_des"] = np.nan
        return candidates

    @property
    def input_preprocessing_specs(self):
        return {}

    def has_sufficient_experiments(
        self,
    ) -> bool:
        return len(self.experiments) >= 3  # type: ignore


STRATEGY_MAP: Dict[Type[data_models.Strategy], Type[Strategy]] = {
    DummyStrategyDataModel: DummyStrategy,
    DummyPredictiveStrategyDataModel: DummyPredictiveStrategy,
}


def map(data_model: data_models.Strategy) -> Strategy:
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
