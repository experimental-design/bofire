from typing import Union

from bofire.data_models.strategies.botorch import BotorchStrategy
from bofire.data_models.strategies.qehvi import QehviStrategy
from bofire.data_models.strategies.qnehvi import QnehviStrategy
from bofire.data_models.strategies.qparego import QparegoStrategy
from bofire.data_models.strategies.random import RandomStrategy
from bofire.data_models.strategies.sobo import (
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.data_models.strategies.strategy import Strategy

AbstractStrategy = Union[
    Strategy,
    BotorchStrategy,
]

AnyStrategy = Union[
    QehviStrategy,
    QnehviStrategy,
    SoboStrategy,
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    QparegoStrategy,
    RandomStrategy,
]

AnyBotorchStrategy = Union[
    QehviStrategy,
    QnehviStrategy,
    SoboStrategy,
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    QparegoStrategy,
]
