from typing import Optional, Type

import bofire.data_models.strategies.api as data_models
from bofire.strategies.mapper_actual import STRATEGY_MAP as ACTUAL_MAP
from bofire.strategies.mapper_meta import STRATEGY_MAP as META_MAP
from bofire.strategies.strategy import Strategy


def register(
    data_model_cls: Type[data_models.Strategy],
    strategy_cls: Optional[Type[Strategy]] = None,
):
    """Register a custom strategy mapping from data model to functional class.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyDataModel)
        class MyStrategy(Strategy):
            ...

        # Direct call form
        register(MyDataModel, MyStrategy)

    Args:
        data_model_cls: The Pydantic data model class.
        strategy_cls: The functional strategy class. If not provided,
            returns a decorator.

    Returns:
        The strategy class (unchanged) when used as a decorator, None otherwise.
    """

    def _register(cls: Type[Strategy]) -> Type[Strategy]:
        ACTUAL_MAP[data_model_cls] = cls

        # Also register with the data model union so Pydantic accepts the type
        data_models.register_strategy(data_model_cls)

        return cls

    if strategy_cls is not None:
        _register(strategy_cls)
        return None

    return _register


def map(data_model: data_models.Strategy) -> Strategy:
    data_cls = data_model.__class__
    if data_cls in META_MAP:
        cls = META_MAP[data_cls]
    else:
        cls = ACTUAL_MAP[data_cls]

    return cls.from_spec(data_model=data_model)
