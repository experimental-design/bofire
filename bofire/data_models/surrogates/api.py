from typing import Union

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.botorch_surrogates import (
    AnyBotorchSurrogate,
    BotorchSurrogates,
)
from bofire.data_models.surrogates.deterministic import LinearDeterministicSurrogate
from bofire.data_models.surrogates.empirical import EmpiricalSurrogate
from bofire.data_models.surrogates.fully_bayesian import SaasSingleTaskGPSurrogate
from bofire.data_models.surrogates.linear import LinearSurrogate
from bofire.data_models.surrogates.mixed_single_task_gp import (
    MixedSingleTaskGPHyperconfig,
    MixedSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.mixed_tanimoto_gp import MixedTanimotoGPSurrogate
from bofire.data_models.surrogates.mlp import (
    ClassificationMLPEnsemble,
    MLPEnsemble,
    RegressionMLPEnsemble,
)
from bofire.data_models.surrogates.polynomial import PolynomialSurrogate
from bofire.data_models.surrogates.random_forest import RandomForestSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.single_task_gp import (
    SingleTaskGPHyperconfig,
    SingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate
from bofire.data_models.surrogates.trainable import MeanAggregation, SumAggregation
from bofire.data_models.surrogates.xgb import XGBoostSurrogate

AbstractSurrogate = Union[Surrogate, BotorchSurrogate, EmpiricalSurrogate, MLPEnsemble]


AnySurrogate = Union[
    EmpiricalSurrogate,
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate,
    MixedTanimotoGPSurrogate,
    ClassificationMLPEnsemble,
    RegressionMLPEnsemble,
    SaasSingleTaskGPSurrogate,
    XGBoostSurrogate,
    LinearSurrogate,
    PolynomialSurrogate,
    TanimotoGPSurrogate,
    LinearDeterministicSurrogate,
]

AnyTrainableSurrogate = Union[
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate,
    MixedTanimotoGPSurrogate,
    ClassificationMLPEnsemble,
    RegressionMLPEnsemble,
    SaasSingleTaskGPSurrogate,
    XGBoostSurrogate,
    LinearSurrogate,
    PolynomialSurrogate,
    TanimotoGPSurrogate,
]

AnyRegressionSurrogate = Union[
    EmpiricalSurrogate,
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate,
    MixedTanimotoGPSurrogate,
    RegressionMLPEnsemble,
    SaasSingleTaskGPSurrogate,
    XGBoostSurrogate,
    LinearSurrogate,
    PolynomialSurrogate,
    TanimotoGPSurrogate,
    LinearDeterministicSurrogate,
]

AnyClassificationSurrogate = ClassificationMLPEnsemble
