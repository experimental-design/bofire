from typing import Union

from bofire.data_models.strategies.doe import DoEStrategy
from bofire.data_models.strategies.factorial import FactorialStrategy
from bofire.data_models.strategies.polytope import PolytopeSampler
from bofire.data_models.strategies.predictives.botorch import LSRBO, BotorchStrategy
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
from bofire.data_models.strategies.stepwise.conditions import (  # noqa: F401
    AlwaysTrueCondition,
    CombiCondition,
    NumberOfExperimentsCondition,
)
from bofire.data_models.strategies.stepwise.stepwise import (  # noqa: F401
    Step,
    StepwiseStrategy,
)
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.strategies.universal_constraint import (
    UniversalConstraintSampler,
)

AbstractStrategy = Union[
    Strategy,
    BotorchStrategy,
    PredictiveStrategy,
    MultiobjectiveStrategy,
]

AnyStrategy = Union[
    SoboStrategy,
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    CustomSoboStrategy,
    QehviStrategy,
    QnehviStrategy,
    QparegoStrategy,
    PolytopeSampler,
    UniversalConstraintSampler,
    RandomStrategy,
    DoEStrategy,
    StepwiseStrategy,
    FactorialStrategy,
    MoboStrategy,
    ShortestPathStrategy,
]

AnyPredictive = Union[
    SoboStrategy,
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    CustomSoboStrategy,
    QehviStrategy,
    QnehviStrategy,
    QparegoStrategy,
    MoboStrategy,
]

AnySampler = Union[PolytopeSampler, UniversalConstraintSampler]


AnyCondition = Union[NumberOfExperimentsCondition, CombiCondition, AlwaysTrueCondition]


AnyLocalSearchConfig = LSRBO
