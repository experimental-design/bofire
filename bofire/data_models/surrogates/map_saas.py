from typing import Literal, Type

from pydantic import Field, PositiveInt, field_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class AdditiveMapSaasSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """GP whose kernel sums several sparse Matern terms, at different sparsity levels.

    The sparse axis-aligned subspace (SAAS) prior on each term pulls most
    lengthscales towards infinity, so only a few inputs are left with any influence,
    and summing terms at different sparsity levels avoids having to commit to how
    many that is. Fitting by maximum a posteriori rather than by sampling makes it
    orders of magnitude cheaper than `FullyBayesianSingleTaskGPSurrogate`, at the
    price of a point estimate of the hyperparameters. Pick it for many inputs and few
    experiments, where only a handful of inputs are expected to drive the response.

    Examples:
        >>> surrogate = AdditiveMapSaasSingleTaskGPSurrogate(
        ...     inputs=inputs, outputs=outputs, n_taus=8
        ... )
    """

    type: Literal["AdditiveMapSaasSingleTaskGPSurrogate"] = (
        "AdditiveMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = Field(
        default=4,
        description="Number of sparsity levels combined in the model. More levels let it "
        "hedge over how many inputs actually matter, at proportionally higher cost.",
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
    """Ensemble of sparse GPs, each assuming a different number of inputs matters.

    Same sparse axis-aligned subspace (SAAS) prior and same Matern-5/2 kernel as
    `AdditiveMapSaasSingleTaskGPSurrogate`, but the sparsity levels are kept as
    separate models and averaged rather than summed, so the spread between them
    feeds into the predicted uncertainty.
    """

    type: Literal["EnsembleMapSaasSingleTaskGPSurrogate"] = (
        "EnsembleMapSaasSingleTaskGPSurrogate"
    )
    n_taus: PositiveInt = Field(
        default=4,
        description="Number of sparsity levels combined in the model. More levels let it "
        "hedge over how many inputs actually matter, at proportionally higher cost.",
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
