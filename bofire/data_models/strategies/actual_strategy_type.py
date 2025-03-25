from typing import Union

from bofire.data_models.strategies.doe import DoEStrategy
from bofire.data_models.strategies.factorial import FactorialStrategy
from bofire.data_models.strategies.fractional_factorial import (
    FractionalFactorialStrategy,
)
from bofire.data_models.strategies.predictives.active_learning import (
    ActiveLearningStrategy,
)
from bofire.data_models.strategies.predictives.enting import EntingStrategy
from bofire.data_models.strategies.predictives.mobo import MoboStrategy
from bofire.data_models.strategies.predictives.multi_fidelity import (
    MultiFidelityStrategy,
)
from bofire.data_models.strategies.predictives.qparego import QparegoStrategy
from bofire.data_models.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiplicativeAdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.data_models.strategies.random import RandomStrategy
from bofire.data_models.strategies.shortest_path import ShortestPathStrategy


ActualStrategy = Union[
    SoboStrategy,
    AdditiveSoboStrategy,
    ActiveLearningStrategy,
    MultiplicativeSoboStrategy,
    MultiplicativeAdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiFidelityStrategy,
    QparegoStrategy,
    EntingStrategy,
    RandomStrategy,
    DoEStrategy,
    FactorialStrategy,
    MoboStrategy,
    ShortestPathStrategy,
    FractionalFactorialStrategy,
]
