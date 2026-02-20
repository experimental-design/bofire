from typing import Any, Literal, Optional, Type

from pydantic import Field, field_validator

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    ContinuousOutput,
    TaskInput,
)
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.data_models.surrogates.scaler import AnyScaler, Normalize, ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class PFNSurrogate(TrainableBotorchSurrogate):
    """Prior-data Fitted Network (PFN) surrogate model.

    PFN is a pre-trained neural network that can be used for Bayesian optimization
    without requiring training on the specific task. The model is loaded from a
    checkpoint URL and makes predictions based on training data context.

    Attributes:
        type: Discriminator for the surrogate type.
        checkpoint_url: URL or path to the pre-trained PFN model checkpoint.
            Defaults to the pfns4bo_hebo model. Can also use ModelPaths enum values:
            - "pfns4bo_hebo": HEBO-style model with more budget and unused features
            - "pfns4bo_bnn": BNN-style model sampled with warp for HPOB
            Or provide a custom URL to a .pt.gz model file.
        batch_first: Whether the batch dimension is the first dimension of input tensors.
            For batch-first, X has shape `batch x seq_len x features`.
            For non-batch-first, X has shape `seq_len x batch x features`.
        constant_model_kwargs: Dictionary of constant keyword arguments that will be
            passed to the model in each forward pass. Use this to configure model-specific
            behavior during inference.
        load_training_checkpoint: If True, loads a training checkpoint as produced by
            the PFNs training code. If False, loads a pre-trained inference model.
        cache_dir: Directory path for caching downloaded models. If None, uses
            /tmp/botorch_pfn_models.
        multivariate: If True, uses MultivariatePFNModel which returns a joint posterior
            over batch inputs. This requires an additional forward pass and approximation.
            If False, uses standard PFNModel with independent predictions.
        scaler: Scaler to use for input features.
        output_scaler: Scaler to use for output targets.

    Note:
        PFN models are pre-trained and do not require fitting in the traditional sense.
        The "fit" operation simply loads the model and stores the training data as context
        for inference.
    """

    type: Literal["PFNSurrogate"] = "PFNSurrogate"

    # Model loading configuration
    checkpoint_url: str = "pfns4bo_hebo"
    load_training_checkpoint: bool = False
    cache_dir: Optional[str] = None

    # Model architecture configuration
    batch_first: bool = False
    multivariate: bool = False

    # Model inference configuration
    constant_model_kwargs: dict[str, Any] = Field(default_factory=dict)
    # num_samples: int = Field(
    #     default=128,
    #     description="Number of samples to draw from the posterior distribution for prediction.",
    # )

    # Scaling configuration
    scaler: AnyScaler = Normalize()
    output_scaler: ScalerEnum = ScalerEnum.STANDARDIZE

    @field_validator("output_scaler")
    @classmethod
    def validate_output_scaler(cls, v: ScalerEnum) -> ScalerEnum:
        """Validate that output_scaler is not LOG or CHAINED_LOG_STANDARDIZE.

        PFN models return variance estimates that are incompatible with LOG transforms,
        as BoTorch's Log transform does not support untransforming variance.

        Args:
            v: The output scaler value to validate.

        Returns:
            The validated output scaler value.

        Raises:
            ValueError: If output_scaler is LOG or CHAINED_LOG_STANDARDIZE.
        """
        if v in (ScalerEnum.LOG, ScalerEnum.CHAINED_LOG_STANDARDIZE):
            raise ValueError(
                f"PFNSurrogate does not support output_scaler={v.name}. "
                "LOG and CHAINED_LOG_STANDARDIZE transforms are incompatible with "
                "PFN's variance predictions. Use STANDARDIZE, NORMALIZE, or IDENTITY instead."
            )
        return v

    @classmethod
    def _default_categorical_encodings(
        cls,
    ) -> dict[Type[CategoricalInput], CategoricalEncodingEnum | Fingerprints]:
        """Override default categorical encodings for PFN models.

        PFN models work better with ordinal encodings instead of one-hot encoding
        to avoid exceeding the pretrained model's feature dimension limits.
        Pretrained PFN checkpoints have fixed input dimensionality constraints.

        Returns:
            Dictionary mapping categorical input types to their encoding strategies.
        """
        return {
            CategoricalInput: CategoricalEncodingEnum.ORDINAL,
            CategoricalMolecularInput: Fingerprints(),
            CategoricalDescriptorInput: CategoricalEncodingEnum.DESCRIPTOR,
            TaskInput: CategoricalEncodingEnum.ORDINAL,
        }

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Check if the output type is implemented for this surrogate.

        Args:
            my_type: The output feature type to check.

        Returns:
            True if the output type is ContinuousOutput, False otherwise.
        """
        return my_type == ContinuousOutput
