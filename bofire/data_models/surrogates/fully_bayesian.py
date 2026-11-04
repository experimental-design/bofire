from typing import Annotated, List, Literal, Type

from pydantic import AfterValidator, Field, field_validator, model_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate
from bofire.data_models.types import make_unique_validator


class FullyBayesianSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Single-task GP whose hyperparameters are sampled by MCMC (NUTS) instead of
    fitted.

    Predictions average over the sampled models, which accounts for the uncertainty
    in the hyperparameters. This is more robust with few data or many inputs than a
    single fit, but much slower.

    Examples:
        >>> FullyBayesianSingleTaskGPSurrogate(
        ...     inputs=inputs, outputs=outputs, model_type="saas"
        ... )
    """

    type: Literal["FullyBayesianSingleTaskGPSurrogate"] = (
        "FullyBayesianSingleTaskGPSurrogate"
    )
    model_type: Literal["linear", "saas", "hvarfner"] = Field(
        default="saas",
        description="The prior: 'saas' a Matern-5/2 kernel whose lengthscales get a "
        "sparsity prior, for many inputs of which only a few matter; 'hvarfner' a "
        "Matern-5/2 kernel with priors scaled to the number of inputs; 'linear' a "
        "linear kernel.",
    )
    warmup_steps: Annotated[int, Field(ge=1)] = Field(
        default=256,
        description="Number of MCMC steps discarded before samples are kept.",
    )
    num_samples: Annotated[int, Field(ge=1)] = Field(
        default=128,
        description="Number of MCMC samples drawn after the warm-up.",
    )
    thinning: Annotated[int, Field(ge=1)] = Field(
        default=16,
        description="Only every n-th sample is kept, so the model averages over "
        "`num_samples / thinning` sets of hyperparameters. At most `num_samples`.",
    )
    features_to_warp: Annotated[
        List[str], AfterValidator(make_unique_validator("Features"))
    ] = Field(
        default=[],
        description="Keys of inputs whose values are warped by a learned monotone "
        "transformation before the kernel, for responses that change much faster in "
        "parts of an input's range. If empty, no input is warped.",
    )

    @model_validator(mode="after")
    def validate_features_to_warp(self):
        input_keys = self.inputs.get_keys()
        for feature in self.features_to_warp:
            if feature not in input_keys:
                raise ValueError(
                    f"Feature '{feature}' in features_to_warp is not a valid input key."
                )
        return self

    @field_validator("thinning")
    @classmethod
    def validate_thinning(cls, thinning, info):
        if info.data["num_samples"] / thinning < 1:
            raise ValueError("`num_samples` has to be larger than `thinning`.")
        return thinning

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return issubclass(my_type, ContinuousOutput)
