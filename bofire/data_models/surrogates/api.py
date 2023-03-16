from typing import Union

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.botorch_surrogates import (  # noqa: F401
    BotorchSurrogates,
)
from bofire.data_models.surrogates.empirical import EmpiricalSurrogate
from bofire.data_models.surrogates.mixed_single_task_gp import (  # noqa: F401
    MixedSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.mlp import MLPEnsemble
from bofire.data_models.surrogates.random_forest import RandomForestSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum  # noqa: F401
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.data_models.surrogates.surrogate import Surrogate

AbstractSurrogate = Union[Surrogate, BotorchSurrogate, EmpiricalSurrogate]


AnyBotorchSurrogate = Union[
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate,
    MLPEnsemble,
]

AnySurrogate = Union[
    EmpiricalSurrogate,
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate,
    MLPEnsemble,
]
