from typing import Dict, Type

from bofire.data_models.surrogates import api as data_models
from bofire.surrogates.empirical import EmpiricalSurrogate
from bofire.surrogates.fully_bayesian import SaasSingleTaskGPSurrogate
from bofire.surrogates.mixed_single_task_gp import MixedSingleTaskGPSurrogate
from bofire.surrogates.mlp import MLPEnsemble
from bofire.surrogates.random_forest import RandomForestSurrogate
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.xgb import XGBoostSurrogate

SURROGATE_MAP: Dict[Type[data_models.Surrogate], Type[Surrogate]] = {
    data_models.EmpiricalSurrogate: EmpiricalSurrogate,
    data_models.RandomForestSurrogate: RandomForestSurrogate,
    data_models.SingleTaskGPSurrogate: SingleTaskGPSurrogate,
    data_models.MixedSingleTaskGPSurrogate: MixedSingleTaskGPSurrogate,
    data_models.MLPEnsemble: MLPEnsemble,
    data_models.SaasSingleTaskGPSurrogate: SaasSingleTaskGPSurrogate,
    data_models.XGBoostSurrogate: XGBoostSurrogate,
    data_models.LinearSurrogate: SingleTaskGPSurrogate,
    data_models.PolynomialSurrogate: SingleTaskGPSurrogate,
    data_models.TanimotoGPSurrogate: SingleTaskGPSurrogate,
}


def map(data_model: data_models.Surrogate) -> Surrogate:
    cls = SURROGATE_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
