from pydantic import Field, model_validator

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum


class TrainableBotorchSurrogate(BotorchSurrogate):
    """BoTorch surrogate fitted to experiments, with its inputs and outputs scaled for
    the model."""

    scaler: AnyScaler = Field(
        default=Normalize(),
        description="How the continuous inputs and the engineered features are "
        "scaled for the model: `Normalize` maps them to [0, 1], `Standardize` to zero "
        "mean and unit variance over the data, and None leaves them as they are.",
    )
    output_scaler: ScalerEnum = Field(
        default=ScalerEnum.STANDARDIZE,
        description="How the outputs are transformed for fitting: STANDARDIZE to zero "
        "mean and unit variance, LOG to their logarithm (outputs must be positive), "
        "CHAINED_LOG_STANDARDIZE to both, IDENTITY not at all. Predictions are "
        "transformed back.",
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
