from typing import Union

from bofire.strategies import random
from bofire.strategies.botorch import qehvi, qparego, sobo

AnyStrategy = Union[
    qehvi.BoTorchQehviStrategy,
    qehvi.BoTorchQnehviStrategy,
    qparego.BoTorchQparegoStrategy,
    sobo.BoTorchSoboStrategy,
    sobo.BoTorchSoboAdditiveStrategy,
    sobo.BoTorchSoboMultiplicativeStrategy,
    random.RandomStrategy,
]
