from typing import Dict, Type

from bofire.data_models.surrogates import api as data_models
from bofire.surrogates.deterministic import (
    CategoricalDeterministicSurrogate,
    LinearDeterministicSurrogate,
)
from bofire.surrogates.empirical import EmpiricalSurrogate
from bofire.surrogates.fully_bayesian import FullyBayesianSingleTaskGPSurrogate
from bofire.surrogates.map_saas import AdditiveMapSaasSingleTaskGPSurrogate
from bofire.surrogates.mixed_single_task_gp import MixedSingleTaskGPSurrogate
from bofire.surrogates.mixed_tanimoto_gp import MixedTanimotoGPSurrogate
from bofire.surrogates.mlp import ClassificationMLPEnsemble, RegressionMLPEnsemble
from bofire.surrogates.multi_task_gp import MultiTaskGPSurrogate
from bofire.surrogates.random_forest import RandomForestSurrogate
from bofire.surrogates.robust_single_task_gp import RobustSingleTaskGPSurrogate
from bofire.surrogates.shape import PiecewiseLinearGPSurrogate
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.xgb import XGBoostSurrogate


SURROGATE_MAP: Dict[Type[data_models.Surrogate], Type[Surrogate]] = {
    data_models.EmpiricalSurrogate: EmpiricalSurrogate,
    data_models.RandomForestSurrogate: RandomForestSurrogate,
    data_models.SingleTaskGPSurrogate: SingleTaskGPSurrogate,
    data_models.RobustSingleTaskGPSurrogate: RobustSingleTaskGPSurrogate,
    data_models.MixedSingleTaskGPSurrogate: MixedSingleTaskGPSurrogate,
    data_models.MixedTanimotoGPSurrogate: MixedTanimotoGPSurrogate,
    data_models.RegressionMLPEnsemble: RegressionMLPEnsemble,
    data_models.ClassificationMLPEnsemble: ClassificationMLPEnsemble,
    data_models.FullyBayesianSingleTaskGPSurrogate: FullyBayesianSingleTaskGPSurrogate,
    data_models.XGBoostSurrogate: XGBoostSurrogate,
    data_models.LinearSurrogate: SingleTaskGPSurrogate,
    data_models.PolynomialSurrogate: SingleTaskGPSurrogate,
    data_models.TanimotoGPSurrogate: SingleTaskGPSurrogate,
    data_models.LinearDeterministicSurrogate: LinearDeterministicSurrogate,
    data_models.MultiTaskGPSurrogate: MultiTaskGPSurrogate,
    data_models.SingleTaskIBNNSurrogate: SingleTaskGPSurrogate,
    data_models.PiecewiseLinearGPSurrogate: PiecewiseLinearGPSurrogate,
    data_models.CategoricalDeterministicSurrogate: CategoricalDeterministicSurrogate,
    data_models.AdditiveMapSaasSingleTaskGPSurrogate: AdditiveMapSaasSingleTaskGPSurrogate,
}


def map(data_model: data_models.Surrogate) -> Surrogate:
    cls = SURROGATE_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
