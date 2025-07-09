from bofire.data_models import unions
from bofire.data_models.acquisition_functions.api import (
    AcquisitionFunction,
    AnyAcquisitionFunction,
)
from bofire.data_models.constraints.api import AnyConstraint, Constraint
from bofire.data_models.dataframes.api import AnyDataFrame, AnyRow
from bofire.data_models.domain.api import Constraints, Domain, Features, Inputs, Outputs
from bofire.data_models.features.api import (
    AnyFeature,
    AnyInput,
    AnyOutput,
    Feature,
    Input,
    Output,
)
from bofire.data_models.kernels.api import AnyKernel, Kernel
from bofire.data_models.molfeatures.api import AnyMolFeatures, MolFeatures
from bofire.data_models.objectives.api import AnyObjective, Objective
from bofire.data_models.outlier_detection.api import (
    AnyOutlierDetection,
    OutlierDetection,
)
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint, Prior
from bofire.data_models.strategies.api import (
    AnyCondition,
    AnyLocalSearchConfig,
    AnyPredictive,
    AnyStrategy,
    PredictiveStrategy,
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
    AnyCondition,
    AnyConstraint,
    AnyFeature,
    AnyKernel,
    AnySurrogate,
    AnyOutlierDetection,
    AnyObjective,
    AnyPrior,
    AnyPriorConstraint,
    AnyStrategy,
    AnyMolFeatures,
    Domain,
    AnyLocalSearchConfig,
    Inputs,
    Outputs,
    Constraints,
]

AnyThing = [model for models in data_model_list for model in unions.to_list(models)]
