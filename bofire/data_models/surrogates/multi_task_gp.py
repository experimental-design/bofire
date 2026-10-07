from typing import Literal, Optional, Type

from pydantic import Field, model_validator

from bofire.data_models.encodings.api import OneHotEncoding, OrdinalEncoding
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import AnyKernel, RBFKernel
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)
from bofire.data_models.priors.lkj import LKJPrior
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MultiTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["MultiTaskGPSurrogate"] = "MultiTaskGPSurrogate"
    kernel: AnyKernel = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: HVARFNER_NOISE_PRIOR())
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default_factory=lambda: GreaterThan(lower_bound=1e-4),
    )
    task_prior: Optional[LKJPrior] = Field(default_factory=lambda: None)

    @classmethod
    def _default_plain_categorical_encodings(cls) -> dict:
        return {
            CategoricalInput: OneHotEncoding(),
            CategoricalTaskInput: OrdinalEncoding(),
        }

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

    @model_validator(mode="after")
    def validate_task_inputs(self):
        if len(self.inputs.get_keys(CategoricalTaskInput)) != 1:
            raise ValueError("Exactly one task input is required for multi-task GPs.")
        task_feature = self.inputs.get(CategoricalTaskInput)[0]
        if not isinstance(
            self.categorical_encodings[task_feature.key], OrdinalEncoding
        ):
            raise ValueError(
                f"The task feature {task_feature.key} has to be encoded as ordinal."
            )
        return self
