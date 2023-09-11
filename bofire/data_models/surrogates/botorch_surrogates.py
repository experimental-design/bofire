from typing import List, Union

from bofire.data_models.surrogates.empirical import EmpiricalSurrogate
from bofire.data_models.surrogates.fully_bayesian import SaasSingleTaskGPSurrogate
from bofire.data_models.surrogates.mixed_single_task_gp import (
    MixedSingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.mlp import MLPEnsemble
from bofire.data_models.surrogates.random_forest import RandomForestSurrogate
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.data_models.surrogates.surrogates import Surrogates
from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate

AnyBotorchSurrogate = Union[
    EmpiricalSurrogate,
    RandomForestSurrogate,
    SingleTaskGPSurrogate,
    MixedSingleTaskGPSurrogate,
    MLPEnsemble,
    SaasSingleTaskGPSurrogate,
    TanimotoGPSurrogate,
]


class BotorchSurrogates(Surrogates):
    """ "List of botorch surrogates.

    Behaves similar to a Surrogate."""

    surrogates: List[AnyBotorchSurrogate]
