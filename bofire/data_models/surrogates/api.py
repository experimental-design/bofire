from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.botorch_surrogates import (
    AnyBotorchSurrogate,
    BotorchSurrogates,
    register_botorch_surrogate,
)
from bofire.data_models.surrogates.deterministic import (
    CategoricalDeterministicSurrogate,
    LinearDeterministicSurrogate,
)
from bofire.data_models.surrogates.empirical import EmpiricalSurrogate
from bofire.data_models.surrogates.fully_bayesian import (
    FullyBayesianSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.map_saas import EnsembleMapSaasSingleTaskGPSurrogate
from bofire.data_models.surrogates.mlp import (
    ClassificationMLPEnsemble,
    MLPEnsemble,
    RegressionMLPEnsemble,
)
from bofire.data_models.surrogates.multi_task_gp import MultiTaskGPSurrogate
from bofire.data_models.surrogates.optimizer import LBFGSB
from bofire.data_models.surrogates.pairwise_gp import PairwiseGPSurrogate
from bofire.data_models.surrogates.random_forest import RandomForestSurrogate
from bofire.data_models.surrogates.scaler import (
    AnyScaler,
    Normalize,
    ScalerEnum,
    Standardize,
)
from bofire.data_models.surrogates.selection import SelectionSurrogate
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate
from bofire.data_models.unions import tagged_union


AnySurrogate = tagged_union(
    EmpiricalSurrogate,
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    ClassificationMLPEnsemble,
    RegressionMLPEnsemble,
    FullyBayesianSingleTaskGPSurrogate,
    TanimotoGPSurrogate,
    LinearDeterministicSurrogate,
    CategoricalDeterministicSurrogate,
    MultiTaskGPSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
    PairwiseGPSurrogate,
    SelectionSurrogate,
)

AnyTrainableSurrogate = tagged_union(
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    ClassificationMLPEnsemble,
    RegressionMLPEnsemble,
    FullyBayesianSingleTaskGPSurrogate,
    TanimotoGPSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
    SelectionSurrogate,
)

AnyRegressionSurrogate = tagged_union(
    EmpiricalSurrogate,
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    RegressionMLPEnsemble,
    FullyBayesianSingleTaskGPSurrogate,
    TanimotoGPSurrogate,
    LinearDeterministicSurrogate,
    MultiTaskGPSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
    SelectionSurrogate,
)

AnyClassificationSurrogate = ClassificationMLPEnsemble
