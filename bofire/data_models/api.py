from bofire.data_models import unions
from bofire.data_models.acquisition_functions.api import (
    AcquisitionFunction,
    AnyAcquisitionFunction,
)
from bofire.data_models.constraints.api import AnyConstraint, Constraint
from bofire.data_models.domain.api import Domain, Features, Inputs, Outputs
from bofire.data_models.features.api import (
    AnyFeature,
    AnyInput,
    AnyOutput,
    Feature,
    Input,
    Output,
)

try:
    # in case of the minimal installation these import are not available
    from bofire.data_models.kernels.api import AnyKernel, Kernel
    from bofire.data_models.molfeatures.api import (  # noqa: F401
        AnyMolFeatures,
        MolFeatures,
    )
    from bofire.data_models.objectives.api import AnyObjective, Objective
    from bofire.data_models.outlier_detection.api import (
        AnyOutlierDetection,
        OutlierDetection,
    )
    from bofire.data_models.priors.api import AnyPrior, Prior
    from bofire.data_models.strategies.api import (
        AnyCondition,
        AnyPredictive,
        AnySampler,
        AnyStrategy,
        PredictiveStrategy,
        SamplerStrategy,
        Strategy,
    )
    from bofire.data_models.surrogates.api import (
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
        AnyOutlierDetection,
        AnyObjective,
        AnyPrior,
        AnyStrategy,
        AnyMolFeatures,
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

AnyThing = [model for models in data_model_list for model in unions.to_list(models)]
