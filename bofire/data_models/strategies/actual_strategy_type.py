from bofire.data_models.strategies.doe import DoEStrategy
from bofire.data_models.strategies.factorial import FactorialStrategy
from bofire.data_models.strategies.fractional_factorial import (
    FractionalFactorialStrategy,
)
from bofire.data_models.strategies.llm import LLMStrategy
from bofire.data_models.strategies.predictives.active_learning import (
    ActiveLearningStrategy,
)
from bofire.data_models.strategies.predictives.enting import EntingStrategy
from bofire.data_models.strategies.predictives.mobo import MoboStrategy
from bofire.data_models.strategies.predictives.multi_fidelity import (
    MultiFidelityVarianceBasedStrategy,
)
from bofire.data_models.strategies.predictives.multi_fidelity_knowledge_gradient import (
    MultiFidelityHVKGStrategy,
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
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.unions import tagged_union


_ACTUAL_STRATEGY_TYPES: list[type[Strategy]] = [
    SoboStrategy,
    AdditiveSoboStrategy,
    ActiveLearningStrategy,
    MultiplicativeSoboStrategy,
    MultiplicativeAdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiFidelityVarianceBasedStrategy,
    MultiFidelityHVKGStrategy,
    QparegoStrategy,
    EntingStrategy,
    RandomStrategy,
    DoEStrategy,
    FactorialStrategy,
    MoboStrategy,
    ShortestPathStrategy,
    FractionalFactorialStrategy,
    LLMStrategy,
]

ActualStrategy = tagged_union(*_ACTUAL_STRATEGY_TYPES)
