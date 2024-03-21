import bofire.data_models.strategies.api as data_models
from bofire.strategies.mapper_actual import STRATEGY_MAP as ACTUAL_MAP
from bofire.strategies.mapper_meta import STRATEGY_MAP as META_MAP
from bofire.strategies.strategy import Strategy


def map(data_model: data_models.Strategy) -> Strategy:
    data_cls = data_model.__class__
    if data_cls in META_MAP:
        cls = META_MAP[data_cls]
    else:
        cls = ACTUAL_MAP[data_cls]

    return cls.from_spec(data_model=data_model)
