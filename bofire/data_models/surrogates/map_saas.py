from typing import Literal, Type

from pydantic import PositiveInt, field_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class TestSurrogate:
    pass


class AdditiveMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Additive MAP SAAS single-task GP

    Maximum-a-posteriori (MAP) version of the sparse axis-aligned subspace
    `FullyBayesianSingleTaskGPSurrogate` with `model_type` equals to "saas".

    Attributes:
        n_taus (PositiveInt): Number of sub-kernels to use in the SAAS model.
    """

    type: Literal["AdditiveMapSaasSingleTaskGPSurrogate"] = (  # type: ignore
        "AdditiveMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = 4

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
    """Ensemble MAP SAAS single-task GP

    Batched ensemble of ``SingleTaskGP``s with the Matern-5/2 kernel and a SAAS prior.

    Attributes:
        n_taus (PositiveInt): Number of sub-kernels to use in the SAAS model.
        output_scaler (ScalerEnum): Scaler for the output transformation.
    """

    type: Literal["EnsembleMapSaasSingleTaskGPSurrogate"] = (  # type: ignore
        "EnsembleMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = 4
    output_scaler: ScalerEnum = ScalerEnum.STANDARDIZE

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
