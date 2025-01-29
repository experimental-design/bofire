from collections.abc import Sequence
from typing import Annotated, Literal, Type

from pydantic import Field

from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalOutput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MLPEnsemble(TrainableBotorchSurrogate):
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


class RegressionMLPEnsemble(MLPEnsemble):
    type: Literal["RegressionMLPEnsemble"] = "RegressionMLPEnsemble"
    final_activation: Literal["identity"] = "identity"
    scaler: ScalerEnum = ScalerEnum.IDENTITY
    output_scaler: ScalerEnum = ScalerEnum.IDENTITY

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))


class ClassificationMLPEnsemble(MLPEnsemble):
    type: Literal["ClassificationMLPEnsemble"] = "ClassificationMLPEnsemble"
    final_activation: Literal["softmax"] = "softmax"
    scaler: Literal[ScalerEnum.IDENTITY] = ScalerEnum.IDENTITY
    output_scaler: Literal[ScalerEnum.IDENTITY] = ScalerEnum.IDENTITY

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(CategoricalOutput))
