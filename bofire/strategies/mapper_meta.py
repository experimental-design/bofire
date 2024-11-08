from typing import Dict, Type

import bofire.data_models.strategies.api as data_models
from bofire.strategies.stepwise.stepwise import StepwiseStrategy
from bofire.strategies.strategy import Strategy


# Meta strategies compositions of other strategies.
STRATEGY_MAP: Dict[Type[data_models.Strategy], Type[Strategy]] = {
    data_models.StepwiseStrategy: StepwiseStrategy,
}


def map(data_model: data_models.Strategy) -> Strategy:
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
