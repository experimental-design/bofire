from typing import Annotated, Literal, Sequence

from pydantic import Field, validator

from bofire.data_models.features.api import CategoricalOutput
from bofire.data_models.surrogates.botorch import BotorchSurrogate
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class MLPClassifierEnsemble(BotorchSurrogate, TrainableSurrogate):
    type: Literal["MLPClassifierEnsemble"] = "MLPClassifierEnsemble"
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

    @validator("outputs")
    def validate_outputs(cls, outputs):
        """validates outputs

        Raises:
            ValueError: if output type is not CategoricalOutput

        Returns:
            List[CategoricalOutput]
        """
        for o in outputs:
            if not isinstance(o, CategoricalOutput):
                raise ValueError("all outputs need to be categorical")
        return outputs
