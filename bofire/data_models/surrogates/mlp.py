from typing import Annotated, Literal, Sequence

from pydantic import Field, validator

from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class MLPEnsemble(BotorchSurrogate, TrainableSurrogate):
    type: Literal["MLPEnsemble"] = "MLPEnsemble"
    n_estimators: Annotated[int, Field(ge=1)] = 5
    hidden_layer_sizes: Sequence = (100,)
    activation: Literal["relu", "logistic", "tanh"] = "relu"
    dropout: Annotated[float, Field(ge=0.0)] = 0.0
    batch_size: Annotated[int, Field(ge=1)] = 10
    n_epochs: Annotated[int, Field(ge=1)] = 200
    lr: Annotated[float, Field(gt=0.0)] = 1e-4
    weight_decay: Annotated[float, Field(ge=0.0)] = 0.0
    subsample_fraction: Annotated[float, Field(gt=0.0)] = 1.0
    shuffle: bool = True
    scaler: ScalerEnum = ScalerEnum.NORMALIZE
    output_scaler: ScalerEnum = ScalerEnum.STANDARDIZE

    @validator("output_scaler")
    def validate_output_scaler(cls, output_scaler):
        """validates that output_scaler is a valid type

        Args:
            output_scaler (ScalerEnum): Scaler used to transform the output

        Raises:
            ValueError: when ScalerEnum.NORMALIZE is used

        Returns:
            ScalerEnum: Scaler used to transform the output
        """
        if output_scaler == ScalerEnum.NORMALIZE:
            raise ValueError("Normalize is not supported as an output transform.")

        return output_scaler
