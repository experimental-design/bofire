from typing import Dict, Type

import bofire.data_models.strategies.api as data_models
from bofire.strategies.qehvi import QehviStrategy
from bofire.strategies.qnehvi import QnehviStrategy
from bofire.strategies.qparego import QparegoStrategy
from bofire.strategies.random import RandomStrategy
from bofire.strategies.sobo import (
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.strategies.strategy import Strategy

STRATEGY_MAP: Dict[Type[data_models.Strategy], Type[Strategy]] = {
    data_models.RandomStrategy: RandomStrategy,
    data_models.SoboStrategy: SoboStrategy,
    data_models.AdditiveSoboStrategy: AdditiveSoboStrategy,
    data_models.MultiplicativeSoboStrategy: MultiplicativeSoboStrategy,
    data_models.QehviStrategy: QehviStrategy,
    data_models.QnehviStrategy: QnehviStrategy,
    data_models.QparegoStrategy: QparegoStrategy,
}


def map(data_model: data_models.Strategy) -> Strategy:
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
