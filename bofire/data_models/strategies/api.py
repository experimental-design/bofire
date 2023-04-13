from typing import Union

from bofire.data_models.strategies.doe import DoEStrategy
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy
from bofire.data_models.strategies.predictives.multiobjective import (
    MultiobjectiveStrategy,
)
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from bofire.data_models.strategies.predictives.qehvi import QehviStrategy
from bofire.data_models.strategies.predictives.qnehvi import QnehviStrategy
from bofire.data_models.strategies.predictives.qparego import QparegoStrategy
from bofire.data_models.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.data_models.strategies.random import RandomStrategy
from bofire.data_models.strategies.samplers.polytope import PolytopeSampler
from bofire.data_models.strategies.samplers.rejection import RejectionSampler
from bofire.data_models.strategies.samplers.sampler import SamplerStrategy
from bofire.data_models.strategies.strategy import Strategy

AbstractStrategy = Union[
    Strategy,
    BotorchStrategy,
    SamplerStrategy,
    PredictiveStrategy,
    MultiobjectiveStrategy,
]

AnyStrategy = Union[
    SoboStrategy,
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    QehviStrategy,
    QnehviStrategy,
    QparegoStrategy,
    PolytopeSampler,
    RejectionSampler,
    RandomStrategy,
    DoEStrategy,
]

AnyPredictive = Union[
    SoboStrategy,
    AdditiveSoboStrategy,
    MultiplicativeSoboStrategy,
    QehviStrategy,
    QnehviStrategy,
    QparegoStrategy,
]

AnySampler = Union[
    PolytopeSampler,
    RejectionSampler,
]
