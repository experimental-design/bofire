from typing import Union

from bofire.data_models.surrogates.scaler import ScalerEnum  # noqa: F401

try:
    from bofire.data_models.surrogates.botorch import BotorchSurrogate
    from bofire.data_models.surrogates.botorch_surrogates import (  # noqa: F401
        AnyBotorchSurrogate,
        BotorchSurrogates,
    )
    from bofire.data_models.surrogates.empirical import EmpiricalSurrogate
    from bofire.data_models.surrogates.fully_bayesian import SaasSingleTaskGPSurrogate
    from bofire.data_models.surrogates.mixed_single_task_gp import (  # noqa: F401
        MixedSingleTaskGPSurrogate,
    )
    from bofire.data_models.surrogates.mlp import MLPEnsemble
    from bofire.data_models.surrogates.random_forest import RandomForestSurrogate
    from bofire.data_models.surrogates.single_task_gp import SingleTaskGPSurrogate
    from bofire.data_models.surrogates.surrogate import Surrogate

    AbstractSurrogate = Union[Surrogate, BotorchSurrogate, EmpiricalSurrogate]

    AnySurrogate = Union[
        EmpiricalSurrogate,
        RandomForestSurrogate,
        SingleTaskGPSurrogate,
        MixedSingleTaskGPSurrogate,
        MLPEnsemble,
        SaasSingleTaskGPSurrogate,
    ]
except ImportError:
    # with the minimal installationwe don't have botorch
    pass
