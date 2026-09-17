from typing import Annotated, List, Literal, Type

from pydantic import AfterValidator, Field, field_validator, model_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate
from bofire.data_models.types import make_unique_validator


class FullyBayesianSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Gaussian process whose hyperparameters are sampled rather than point-fitted.

    Averaging predictions over the sampled hyperparameters gives more honest uncertainty
    than fitting one best value, which matters most when data is scarce. Fitting runs
    MCMC and so costs considerably more than the other GPs.
    """

    type: Literal["FullyBayesianSingleTaskGPSurrogate"] = (
        "FullyBayesianSingleTaskGPSurrogate"
    )
    model_type: Literal["linear", "saas", "hvarfner"] = Field(
        default="saas",
        description="Which prior structure to sample under. `saas` puts strong "
        "sparsity on the lengthscales, so it suits high-dimensional problems where few "
        "inputs matter.",
    )
    warmup_steps: Annotated[int, Field(ge=1)] = Field(
        default=256,
        description="MCMC steps discarded before sampling starts, allowing the chain "
        "to reach its stationary distribution.",
    )
    num_samples: Annotated[int, Field(ge=1)] = Field(
        default=128,
        description="MCMC steps run after warm-up. Only every `thinning`-th of them "
        "is retained.",
    )
    thinning: Annotated[int, Field(ge=1)] = Field(
        default=16,
        description="Retain only every nth of the sampled steps, reducing the "
        "correlation between consecutive draws. The model ends up with "
        "`num_samples / thinning` hyperparameter sets.",
    )
    features_to_warp: Annotated[
        List[str], AfterValidator(make_unique_validator("Features"))
    ] = Field(
        default=[],
        description="Keys of the inputs to pass through a learned monotonic warping, "
        "which lets the model fit a response that varies faster in one part of an "
        "input's range than another.",
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
        return isinstance(my_type, type(ContinuousOutput))
