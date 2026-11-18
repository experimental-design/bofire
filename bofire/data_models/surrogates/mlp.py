from collections.abc import Sequence
from typing import Annotated, Literal, Type

from pydantic import Field, PositiveInt

from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalOutput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.scaler import AnyScaler, ScalerEnum, Standardize
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MLPEnsemble(TrainableBotorchSurrogate):
    """Ensemble of small neural networks (multilayer perceptrons).

    Every network is trained on a resample of the experiments drawn with
    replacement; predictions average over the networks, and their spread serves as
    the uncertainty. Suited to larger data sets than a Gaussian process.
    """

    type: Literal["MLPEnsemble"] = "MLPEnsemble"
    n_estimators: Annotated[int, Field(ge=1)] = Field(
        default=5,
        description="Number of networks in the ensemble. More networks give a more "
        "reliable uncertainty at a higher cost.",
    )
    hidden_layer_sizes: Sequence[PositiveInt] = Field(
        default=(100,),
        description="Number of units of each hidden layer, from input to output.",
    )
    activation: Literal["relu", "logistic", "tanh"] = Field(
        default="relu",
        description="Activation function of the hidden layers.",
    )
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)] = Field(
        default=0.0,
        description="Probability with which a hidden unit is left out in each "
        "training step, as regularization; 0 for none.",
    )
    batch_size: Annotated[int, Field(ge=1)] = Field(
        default=10,
        description="Number of experiments per training step.",
    )
    n_epochs: Annotated[int, Field(ge=1)] = Field(
        default=200,
        description="Number of passes over the training data.",
    )
    lr: Annotated[float, Field(gt=0.0)] = Field(
        default=1e-4,
        description="Learning rate of the Adam optimizer.",
    )
    weight_decay: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-4,
        description="Strength of the L2 regularization of the weights; 0 for none.",
    )
    subsample_fraction: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Size of each network's resample, as a fraction of the "
        "experiments; it is drawn with replacement, so values above 1 are possible.",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether the order of the experiments is shuffled in every "
        "epoch.",
    )


class RegressionMLPEnsemble(MLPEnsemble):
    """Ensemble of neural networks for a continuous output, trained on the mean
    absolute error.

    Examples:
        >>> RegressionMLPEnsemble(
        ...     inputs=inputs, outputs=outputs, hidden_layer_sizes=(32, 32)
        ... )
    """

    type: Literal["RegressionMLPEnsemble"] = "RegressionMLPEnsemble"
    scaler: AnyScaler = Field(
        default=Standardize(),
        description="How the continuous inputs and the engineered features are "
        "scaled for the networks; None leaves them as they are.",
    )
    output_scaler: ScalerEnum = Field(
        default=ScalerEnum.IDENTITY,
        description="How the output is transformed for training.",
    )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return issubclass(my_type, ContinuousOutput)


class ClassificationMLPEnsemble(MLPEnsemble):
    """Ensemble of neural networks for a categorical output, predicting the
    probability of every category; trained on the cross-entropy.

    Examples:
        >>> ClassificationMLPEnsemble(inputs=inputs, outputs=outputs)
    """

    type: Literal["ClassificationMLPEnsemble"] = "ClassificationMLPEnsemble"
    scaler: AnyScaler = Field(
        default=Standardize(),
        description="How the continuous inputs and the engineered features are "
        "scaled for the networks; None leaves them as they are.",
    )
    output_scaler: Literal[ScalerEnum.IDENTITY] = Field(
        default=ScalerEnum.IDENTITY,
        description="The categorical output is not transformed.",
    )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return issubclass(my_type, CategoricalOutput)
