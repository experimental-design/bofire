from typing import Union

from bofire.data_models.strategies.actual_strategy_type import ActualStrategy
from bofire.data_models.strategies.doe import (
    AnyDoEOptimalityCriterion,
    AnyOptimalityCriterion,
    AOptimalityCriterion,
    DoEStrategy,
    DOptimalityCriterion,
    EOptimalityCriterion,
    GOptimalityCriterion,
    KOptimalityCriterion,
    SpaceFillingCriterion,
)
from bofire.data_models.strategies.factorial import FactorialStrategy
from bofire.data_models.strategies.fractional_factorial import (
    FractionalFactorialStrategy,
)
from bofire.data_models.strategies.meta_strategy_type import MetaStrategy
from bofire.data_models.strategies.predictives.acqf_optimization import (
    LSRBO,
    AcquisitionOptimizer,
    AnyAcqfOptimizer,
    BotorchOptimizer,
    GeneticAlgorithmOptimizer,
)
from bofire.data_models.strategies.predictives.active_learning import (
    ActiveLearningStrategy,
)
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy
from bofire.data_models.strategies.predictives.enting import EntingStrategy
from bofire.data_models.strategies.predictives.mobo import (
    AbsoluteMovingReferenceValue,
    ExplicitReferencePoint,
    FixedReferenceValue,
    MoboStrategy,
    RelativeMovingReferenceValue,
    RelativeToMaxMovingReferenceValue,
)
from bofire.data_models.strategies.predictives.multi_fidelity import (
    MultiFidelityStrategy,
)
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
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
from bofire.data_models.strategies.stepwise.conditions import (
    AlwaysTrueCondition,
    AnyCondition,
    CombiCondition,
    FeasibleExperimentCondition,
    NumberOfExperimentsCondition,
)
from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy
from bofire.data_models.strategies.strategy import Strategy
from bofire.data_models.transforms.api import (
    AnyTransform,
    DropDataTransform,
    ManipulateDataTransform,
)


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
    MultiplicativeAdditiveSoboStrategy,
    CustomSoboStrategy,
    QparegoStrategy,
    EntingStrategy,
    MoboStrategy,
]

AnyLocalSearchConfig = LSRBO
