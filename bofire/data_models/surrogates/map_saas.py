from typing import Literal, Type

from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.optimizer import LBFGSBFitted
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class EnsembleMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate, LBFGSBFitted):
    """Batched ensemble of single-task GPs with a Matern-5/2 kernel and a sparse
    axis-aligned subspace (SAAS) prior, as BoTorch's `EnsembleMapSaasSingleTaskGP`
    (Daulton et al., BONSAI, 2026).

    Every member of the ensemble has a sparsity parameter of its own, drawn once and
    then kept fixed, not learned; the other hyperparameters are fitted by maximizing
    the marginal likelihood, and the predictions form a mixture over the members. Outputs can be standardized or left
    as they are, but not log-transformed.

    Examples:
        >>> EnsembleMapSaasSingleTaskGPSurrogate(
        ...     inputs=inputs, outputs=outputs, n_taus=4
        ... )
    """

    type: Literal["EnsembleMapSaasSingleTaskGPSurrogate"] = (
        "EnsembleMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = Field(
        default=4,
        description="Number of members of the ensemble, each with a sparsity strength "
        "of its own, drawn when the model is built.",
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
        return issubclass(my_type, ContinuousOutput)
