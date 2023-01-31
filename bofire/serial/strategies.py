from typing import Union

from bofire.strategies.botorch.qehvi import BoTorchQehviStrategy, BoTorchQnehviStrategy
from bofire.strategies.botorch.qparego import BoTorchQparegoStrategy
from bofire.strategies.botorch.sobo import (
    BoTorchSoboAdditiveStrategy,
    BoTorchSoboMultiplicativeStrategy,
    BoTorchSoboStrategy,
)
from bofire.strategies.random import RandomStrategy

AnyStrategy = Union[
    BoTorchQehviStrategy,
    BoTorchQnehviStrategy,
    BoTorchQparegoStrategy,
    BoTorchSoboStrategy,
    BoTorchSoboAdditiveStrategy,
    BoTorchSoboMultiplicativeStrategy,
    RandomStrategy,
]
