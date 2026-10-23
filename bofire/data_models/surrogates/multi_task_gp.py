from typing import Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.encodings.api import OneHotEncoding, OrdinalEncoding
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import AnyKernel, RBFKernel
from bofire.data_models.likelihoods.api import HVARFNER_LIKELIHOOD, GaussianLikelihood
from bofire.data_models.means.api import AnyMean, ConstantMean
from bofire.data_models.priors.api import HVARFNER_LENGTHSCALE_PRIOR, GreaterThan
from bofire.data_models.surrogates.optimizer import LBFGSBFitted
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MultiTaskGPSurrogate(TrainableBotorchSurrogate, LBFGSBFitted):
    """Gaussian process over several related tasks, which learns how strongly they are
    correlated and transfers what it learns about one task to the others.

    The tasks are the categories of the one `CategoricalTaskInput`, e.g. a cheap
    simulation and the real experiment, or a previous and the current campaign. The
    covariance is the kernel over the other inputs times a learned correlation between
    the tasks (intrinsic coregionalization). Every task has a mean and a noise level of
    its own, so tasks may differ in offset and in measurement precision. Predictions
    for a task use the data of all tasks, weighted by how strongly they correlate with
    it.

    Examples:
        >>> MultiTaskGPSurrogate(
        ...     inputs=Inputs(features=[
        ...         ContinuousInput(key="x", bounds=(0, 1)),
        ...         CategoricalTaskInput(key="task", categories=["simulation", "experiment"]),
        ...     ]),
        ...     outputs=Outputs(features=[ContinuousOutput(key="y")]),
        ... )
    """

    type: Literal["MultiTaskGPSurrogate"] = "MultiTaskGPSurrogate"
    kernel: AnyKernel = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
            lengthscale_constraint=GreaterThan(lower_bound=2.5e-2),
        ),
        description="Kernel over all inputs except the task input; the correlation "
        "between the tasks is learned separately.",
    )
    mean: AnyMean = Field(
        default=ConstantMean(),
        description="Prior mean function, applied per task: every task gets a mean "
        "of its own, each with this prior and these bounds.",
    )
    likelihood: GaussianLikelihood = Field(
        default=HVARFNER_LIKELIHOOD(),
        description="Model of the measurement noise, applied per task: every task "
        "gets a noise level of its own, each with this prior and these bounds.",
    )

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
