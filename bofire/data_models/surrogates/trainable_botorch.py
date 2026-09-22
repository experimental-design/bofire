from pydantic import Field, model_validator

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


# reused by the GP surrogates, which must redeclare these fields because their defaults
# differ and pydantic cannot override a default without redeclaring
NOISE_PRIOR_DESCRIPTION = (
    "Prior over the observation noise, which sets how much of the spread in the data "
    "the model attributes to measurement error rather than to the response."
)
NOISE_CONSTRAINT_DESCRIPTION = (
    "Bounds the observation noise is restricted to during fitting. A positive lower "
    "bound keeps the fit numerically stable."
)
HYPERCONFIG_DESCRIPTION = (
    "Configuration of a hyperparameter optimization for this surrogate. Carrying it "
    "does not run anything; without it, no hyperparameter optimization is possible."
)
# also used by PairwiseGPSurrogate, which scales its inputs but has no observed output
# to scale, and so does not inherit from TrainableBotorchSurrogate
SCALER_DESCRIPTION = (
    "How the inputs are rescaled before fitting. Set to null to leave them as they are."
)


class TrainableBotorchSurrogate(BotorchSurrogate, TrainableSurrogate):
    """BoTorch based surrogate fitted to the experiments.

    Fitting is often scale-sensitive, so the inputs and the output are rescaled by
    default before the fit, and the output scaling is undone when predicting.
    """

    scaler: AnyScaler = Field(
        default_factory=Normalize,
        description=SCALER_DESCRIPTION,
    )
    output_scaler: ScalerEnum = Field(
        default=ScalerEnum.STANDARDIZE,
        description="How the outputs are rescaled before fitting, and undone when "
        "predicting.",
    )

    @model_validator(mode="after")
    def validate_scaler_features(self):
        if self.scaler and len(self.scaler.features) > 0:
            known_keys = self.inputs.get_keys() + self.engineered_features.get_keys()
            missing_features = list(set(self.scaler.features) - set(known_keys))
            if missing_features:
                raise ValueError(
                    f"The following features are missing in inputs: {missing_features}"
                )
        return self
