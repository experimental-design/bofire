from pydantic import Field, model_validator

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class TrainableBotorchSurrogate(BotorchSurrogate, TrainableSurrogate):
    scaler: AnyScaler = Field(default_factory=Normalize)
    output_scaler: ScalerEnum = ScalerEnum.STANDARDIZE

    @model_validator(mode="after")
    def validate_scaler_features(self):
        if self.scaler and len(self.scaler.features) > 0:
            missing_features = list(
                set(self.scaler.features) - set(self.inputs.get_keys())
            )
            if missing_features:
                raise ValueError(
                    f"The following features are missing in inputs: {missing_features}"
                )
        return self
