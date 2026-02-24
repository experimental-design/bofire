from typing import List, Type, Union

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
from bofire.data_models.strategies.strategy import Strategy


_ACTUAL_STRATEGY_TYPES: List[Type[Strategy]] = [
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

ActualStrategy = Union[tuple(_ACTUAL_STRATEGY_TYPES)]


def register_strategy(data_model_cls: Type[Strategy]) -> None:
    """Register a custom strategy type so it is accepted in ActualStrategy fields.

    This appends the type to the internal registry, rebuilds the
    ``ActualStrategy`` union, and calls ``model_rebuild`` on the
    ``Step`` and ``StepwiseStrategy`` models so that Pydantic accepts the
    new type.

    Args:
        data_model_cls: A concrete subclass of ``Strategy``.
    """
    global ActualStrategy
    if data_model_cls in _ACTUAL_STRATEGY_TYPES:
        return
    _ACTUAL_STRATEGY_TYPES.append(data_model_cls)
    ActualStrategy = Union[tuple(_ACTUAL_STRATEGY_TYPES)]

    from bofire.data_models._register_utils import patch_field
    from bofire.data_models.strategies.stepwise.stepwise import Step, StepwiseStrategy

    patch_field(Step, "strategy_data", ActualStrategy)
    Step.model_rebuild(force=True)
    StepwiseStrategy.model_rebuild(force=True)
