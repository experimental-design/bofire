from bofire.data_models import unions
from bofire.data_models.acquisition_functions.api import (  # noqa: F401
    AcquisitionFunction,
    AnyAcquisitionFunction,
)
from bofire.data_models.constraints.api import AnyConstraint, Constraint  # noqa: F401
from bofire.data_models.domain.api import (  # noqa: F401
    Domain,
    Features,
    Inputs,
    Outputs,
)
from bofire.data_models.features.api import (  # noqa: F401
    AnyFeature,
    AnyInput,
    AnyOutput,
    Feature,
    Input,
    Output,
)
from bofire.data_models.kernels.api import AnyKernel, Kernel  # noqa: F401
from bofire.data_models.objectives.api import AnyObjective, Objective  # noqa: F401
from bofire.data_models.priors.api import AnyPrior, Prior  # noqa: F401
from bofire.data_models.samplers.api import AnySampler, Sampler  # noqa: F401
from bofire.data_models.strategies.api import AnyStrategy, Strategy  # noqa: F401
from bofire.data_models.surrogates.api import (  # noqa: F401
    AnyBotorchSurrogate,
    AnySurrogate,
    BotorchSurrogate,
    Surrogate,
)

AnyThing = [
    model
    for models in [
        AnyAcquisitionFunction,
        AnyConstraint,
        AnyFeature,
        AnyKernel,
        AnySurrogate,
        AnyObjective,
        AnyPrior,
        AnySampler,
        AnyStrategy,
        Domain,
    ]
    for model in unions.to_list(models)
]
