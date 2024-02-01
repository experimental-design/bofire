from typing import Dict, Type

import bofire.data_models.strategies.api as data_models
from bofire.strategies.doe_strategy import DoEStrategy
from bofire.strategies.factorial import FactorialStrategy
from bofire.strategies.polytope import PolytopeSampler
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.strategies.predictives.mobo import MoboStrategy
from bofire.strategies.predictives.predictive import PredictiveStrategy
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
from bofire.strategies.stepwise.stepwise import StepwiseStrategy
from bofire.strategies.strategy import Strategy
from bofire.strategies.universal_constraint import UniversalConstraintSampler

STRATEGY_MAP: Dict[Type[data_models.Strategy], Type[Strategy]] = {
    data_models.RandomStrategy: RandomStrategy,
    data_models.SoboStrategy: SoboStrategy,
    data_models.AdditiveSoboStrategy: AdditiveSoboStrategy,
    data_models.MultiplicativeSoboStrategy: MultiplicativeSoboStrategy,
    data_models.CustomSoboStrategy: CustomSoboStrategy,
    data_models.QehviStrategy: QehviStrategy,
    data_models.QnehviStrategy: QnehviStrategy,
    data_models.QparegoStrategy: QparegoStrategy,
    data_models.PolytopeSampler: PolytopeSampler,
    data_models.UniversalConstraintSampler: UniversalConstraintSampler,
    data_models.DoEStrategy: DoEStrategy,
    data_models.StepwiseStrategy: StepwiseStrategy,
    data_models.FactorialStrategy: FactorialStrategy,
    data_models.MoboStrategy: MoboStrategy,
    data_models.ShortestPathStrategy: ShortestPathStrategy,
}


def map(data_model: data_models.Strategy) -> Strategy:
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
