from typing import Type

import bofire.data_models.strategies.api as data_models
from bofire.strategies.mapper_actual import STRATEGY_MAP as ACTUAL_MAP
from bofire.strategies.mapper_meta import STRATEGY_MAP as META_MAP
from bofire.strategies.strategy import Strategy


def register(
    data_model_cls: Type[data_models.Strategy],
    strategy_cls: Type[Strategy],
    meta: bool = False,
) -> None:
    """Register a custom strategy mapping from data model to functional class.

    Args:
        data_model_cls: The Pydantic data model class.
        strategy_cls: The functional strategy class.
        meta: If True, register as a meta strategy (e.g. compositions of strategies).
    """
    if meta:
        META_MAP[data_model_cls] = strategy_cls
    else:
        ACTUAL_MAP[data_model_cls] = strategy_cls


def map(data_model: data_models.Strategy) -> Strategy:
    data_cls = data_model.__class__
    if data_cls in META_MAP:
        cls = META_MAP[data_cls]
    else:
        cls = ACTUAL_MAP[data_cls]

    return cls.from_spec(data_model=data_model)
