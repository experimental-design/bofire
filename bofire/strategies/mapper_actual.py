from typing import Dict, Type

import bofire.data_models.strategies.api as data_models
from bofire.strategies.doe_strategy import DoEStrategy
from bofire.strategies.factorial import FactorialStrategy
from bofire.strategies.fractional_factorial import FractionalFactorialStrategy
from bofire.strategies.predictives.active_learning import ActiveLearningStrategy
from bofire.strategies.predictives.enting import EntingStrategy
from bofire.strategies.predictives.mobo import MoboStrategy
from bofire.strategies.predictives.multi_fidelity import MultiFidelityStrategy
from bofire.strategies.predictives.qparego import QparegoStrategy
from bofire.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiplicativeAdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.strategies.random import RandomStrategy
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.strategies.strategy import Strategy


STRATEGY_MAP: Dict[Type[data_models.Strategy], Type[Strategy]] = {
    data_models.ActiveLearningStrategy: ActiveLearningStrategy,
    data_models.RandomStrategy: RandomStrategy,
    data_models.SoboStrategy: SoboStrategy,
    data_models.AdditiveSoboStrategy: AdditiveSoboStrategy,
    data_models.MultiplicativeSoboStrategy: MultiplicativeSoboStrategy,
    data_models.MultiplicativeAdditiveSoboStrategy: MultiplicativeAdditiveSoboStrategy,
    data_models.CustomSoboStrategy: CustomSoboStrategy,
    data_models.MultiFidelityStrategy: MultiFidelityStrategy,
    data_models.QparegoStrategy: QparegoStrategy,
    data_models.EntingStrategy: EntingStrategy,
    data_models.DoEStrategy: DoEStrategy,
    data_models.FactorialStrategy: FactorialStrategy,
    data_models.MoboStrategy: MoboStrategy,
    data_models.ShortestPathStrategy: ShortestPathStrategy,
    data_models.FractionalFactorialStrategy: FractionalFactorialStrategy,
}


def map(data_model: data_models.Strategy) -> Strategy:
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
