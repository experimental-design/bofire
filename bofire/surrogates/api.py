from bofire.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.surrogates.deterministic import LinearDeterministicSurrogate
from bofire.surrogates.empirical import EmpiricalSurrogate
from bofire.surrogates.map_saas import AdditiveMapSaasSingleTaskGPSurrogate
from bofire.surrogates.mapper import map
from bofire.surrogates.mlp import (
    ClassificationMLPEnsemble,
    MLPEnsemble,
    RegressionMLPEnsemble,
)
from bofire.surrogates.multi_task_gp import MultiTaskGPSurrogate
from bofire.surrogates.random_forest import RandomForestSurrogate
from bofire.surrogates.shape import PiecewiseLinearGPSurrogate
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.values import PredictedValue
