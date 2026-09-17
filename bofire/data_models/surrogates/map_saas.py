from typing import Literal, Type

from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class AdditiveMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Maximum-a-posteriori approximation of the fully Bayesian SAAS model.

    Pick it for many inputs and few experiments, where only a handful of inputs are
    expected to drive the response, and the sampling that
    `FullyBayesianSingleTaskGPSurrogate` with `model_type="saas"` does is too expensive.
    """

    type: Literal["AdditiveMapSaasSingleTaskGPSurrogate"] = (
        "AdditiveMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = Field(
        default=4,
        description="Number of sparse Matern kernels that are summed up, each at its "
        "own sparsity level.",
    )

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))


class EnsembleMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Maximum-a-posteriori approximation of the fully Bayesian SAAS model.

    Approximates the same model as `AdditiveMapSaasSingleTaskGPSurrogate` by a different
    mechanism -- the sparsity levels are kept as separate models and their predictions
    mixed, rather than summed into one kernel. This is the preferred of the two.
    """

    type: Literal["EnsembleMapSaasSingleTaskGPSurrogate"] = (
        "EnsembleMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = Field(
        default=4,
        description="Number of sparse Matern kernels in the ensemble, each at its own "
        "sparsity level.",
    )
    output_scaler: ScalerEnum = Field(
        default=ScalerEnum.STANDARDIZE,
        description="How the outputs are rescaled before fitting. The log-based "
        "scalers are not supported here.",
    )

    @field_validator("output_scaler")
    @classmethod
    def validate_output_scaler(cls, output_scaler):
        """Validates that output_scaler is a valid type

        Args:
            output_scaler (ScalerEnum): Scaler used to transform the output

        Raises:
            ValueError: when ScalerEnum.LOG or ScalerEnum.CHAINED_LOG_STANDARDIZE is used

        Returns:
            ScalerEnum: Scaler used to transform the output

        """
        if output_scaler in [ScalerEnum.LOG, ScalerEnum.CHAINED_LOG_STANDARDIZE]:
            raise ValueError(
                "LOG and CHAINED_LOG_STANDARDIZE are not supported as output transforms for EnsembleMapSaasSingleTaskGPSurrogate."
            )
        return output_scaler

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
