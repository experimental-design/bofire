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

try:
    # in case of the minimal installation these import are not available
    from bofire.data_models.kernels.api import AnyKernel, Kernel  # noqa: F401
    from bofire.data_models.objectives.api import AnyObjective, Objective  # noqa: F401
    from bofire.data_models.priors.api import AnyPrior, Prior  # noqa: F401
    from bofire.data_models.strategies.api import (  # noqa: F401
        AnyPredictive,
        AnySampler,
        AnyStrategy,
        PredictiveStrategy,
        SamplerStrategy,
        Strategy,
    )
    from bofire.data_models.surrogates.api import (  # noqa: F401
        AnyBotorchSurrogate,
        AnySurrogate,
        BotorchSurrogate,
        Surrogate,
    )

    data_model_list = [
        AnyAcquisitionFunction,
        AnyConstraint,
        AnyFeature,
        AnyKernel,
        AnySurrogate,
        AnyObjective,
        AnyPrior,
        AnyStrategy,
        Domain,
    ]
except ImportError:
    data_model_list = [
        AnyAcquisitionFunction,
        AnyConstraint,
        AnyFeature,
        Domain,
    ]

AnyThing = [model for models in data_model_list for model in unions.to_list(models)]
