from typing import Dict, Tuple, Type

import pandas as pd

import bofire.data_models.strategies.api as data_models
import bofire.strategies.stepwise.conditions as conditions
from bofire.data_models.strategies.api import Step
from bofire.data_models.strategies.api import StepwiseStrategy as data_model
from bofire.strategies.doe_strategy import DoEStrategy
from bofire.strategies.factorial import FactorialStrategy
from bofire.strategies.predictives.mobo import MoboStrategy
from bofire.strategies.predictives.qehvi import QehviStrategy
from bofire.strategies.predictives.qnehvi import QnehviStrategy
from bofire.strategies.predictives.qparego import QparegoStrategy
from bofire.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.strategies.random import RandomStrategy
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.strategies.space_filling import SpaceFillingStrategy
from bofire.strategies.strategy import Strategy

# we have to duplicate the map functionality due to prevent circular imports
STRATEGY_MAP: Dict[Type[data_models.Strategy], Type[Strategy]] = {
    data_models.RandomStrategy: RandomStrategy,
    data_models.SoboStrategy: SoboStrategy,
    data_models.AdditiveSoboStrategy: AdditiveSoboStrategy,
    data_models.MultiplicativeSoboStrategy: MultiplicativeSoboStrategy,
    data_models.CustomSoboStrategy: CustomSoboStrategy,
    data_models.QehviStrategy: QehviStrategy,
    data_models.QnehviStrategy: QnehviStrategy,
    data_models.QparegoStrategy: QparegoStrategy,
    data_models.SpaceFillingStrategy: SpaceFillingStrategy,
    data_models.DoEStrategy: DoEStrategy,
    data_models.FactorialStrategy: FactorialStrategy,
    data_models.MoboStrategy: MoboStrategy,
    data_models.ShortestPathStrategy: ShortestPathStrategy,
}


def map(data_model: data_models.Strategy) -> Strategy:
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)


class StepwiseStrategy(Strategy):
    def __init__(self, data_model: data_model, **kwargs):
        super().__init__(data_model, **kwargs)
        self.steps = data_model.steps

    def has_sufficient_experiments(self) -> bool:
        return True

    def _get_step(self) -> Tuple[int, Step]:  # type: ignore
        for i, step in enumerate(self.steps):
            condition = conditions.map(step.condition)
            if condition.evaluate(self.domain, experiments=self.experiments):
                return i, step
        raise ValueError("No condition could be satisfied.")

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        # we have to decide here w
        istep, step = self._get_step()
        if (step.max_parallelism > 0) and (candidate_count > step.max_parallelism):
            raise ValueError(
                f"Maximum number of candidates for step {istep} is {step.max_parallelism}."
            )
        # map it
        strategy = map(step.strategy_data)
        # tell the experiments
        if self.num_experiments > 0:
            strategy.tell(experiments=self.experiments)
        # tell pending
        if self.num_candidates > 0:
            strategy.set_candidates(self.candidates)
        # ask and return
        return strategy.ask(candidate_count=candidate_count)
