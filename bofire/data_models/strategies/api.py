from typing import Union

from bofire.data_models.strategies.actual_strategy_type import ActualStrategy
from bofire.data_models.strategies.doe import DoEStrategy
from bofire.data_models.strategies.factorial import FactorialStrategy
from bofire.data_models.strategies.fractional_factorial import (
    FractionalFactorialStrategy,
)
from bofire.data_models.strategies.meta_strategy_type import MetaStrategy
from bofire.data_models.strategies.predictives.active_learning import (
    ActiveLearningStrategy,
)
from bofire.data_models.strategies.predictives.botorch import LSRBO, BotorchStrategy
from bofire.data_models.strategies.predictives.enting import EntingStrategy
from bofire.data_models.strategies.predictives.mobo import MoboStrategy
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from bofire.data_models.strategies.predictives.qehvi import QehviStrategy
from bofire.data_models.strategies.predictives.qnehvi import QnehviStrategy
from bofire.data_models.strategies.predictives.qparego import QparegoStrategy
from bofire.data_models.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.data_models.strategies.random import RandomStrategy
from bofire.data_models.strategies.shortest_path import ShortestPathStrategy
from bofire.data_models.strategies.space_filling import SpaceFillingStrategy
from bofire.data_models.strategies.stepwise.conditions import (
    AlwaysTrueCondition,
    AnyCondition,
    CombiCondition,
    NumberOfExperimentsCondition,
)
from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.transforms.api import AnyTransform, DropDataTransform


AbstractStrategy = Union[
    Strategy,
    BotorchStrategy,
    PredictiveStrategy,
    MultiobjectiveStrategy,
]

AnyStrategy = Union[ActualStrategy, MetaStrategy]

AnyPredictive = Union[
    SoboStrategy,
    ActiveLearningStrategy,
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    CustomSoboStrategy,
    QehviStrategy,
    QnehviStrategy,
    QparegoStrategy,
    EntingStrategy,
    MoboStrategy,
]


AnyLocalSearchConfig = LSRBO
