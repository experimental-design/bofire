from typing import Literal, Optional, Type

import pandas as pd
from pydantic import Field, model_validator

from bofire.data_models.domain.api import Inputs
from bofire.data_models.encodings.api import OneHotEncoding, OrdinalEncoding
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import AnyKernel, MaternKernel, RBFKernel
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    MBO_LENGTHSCALE_PRIOR,
    MBO_NOISE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)
from bofire.data_models.priors.lkj import LKJPrior
from bofire.data_models.surrogates.botorch import KERNEL_DESCRIPTION
from bofire.data_models.surrogates.trainable import (
    HYPERCONFIG_INPUTS_DESCRIPTION,
    HYPERSTRATEGY_DESCRIPTION,
    Hyperconfig,
)
from bofire.data_models.surrogates.trainable_botorch import (
    HYPERCONFIG_DESCRIPTION,
    NOISE_CONSTRAINT_DESCRIPTION,
    NOISE_PRIOR_DESCRIPTION,
    TrainableBotorchSurrogate,
)


class MultiTaskGPHyperconfig(Hyperconfig):
    """Hyperparameter optimization config for a multi-task GP.

    Optimizes over the kernel, the prior family and whether the lengthscale is
    per-input.
    """

    type: Literal["MultiTaskGPHyperconfig"] = "MultiTaskGPHyperconfig"
    inputs: Inputs = Field(
        default=Inputs(
            features=[
                CategoricalInput(
                    key="kernel",
                    categories=["rbf", "matern_1.5", "matern_2.5"],
                ),
                CategoricalInput(
                    key="prior", categories=["mbo", "threesix", "hvarfner"]
                ),
                CategoricalInput(key="ard", categories=["True", "False"]),
            ],
        ),
        description=HYPERCONFIG_INPUTS_DESCRIPTION,
    )
    hyperstrategy: Literal[
        "FractionalFactorialStrategy", "SoboStrategy", "RandomStrategy"
    ] = Field(
        default="FractionalFactorialStrategy", description=HYPERSTRATEGY_DESCRIPTION
    )

    @staticmethod
    def _update_hyperparameters(
        surrogate_data: "MultiTaskGPSurrogate",
        hyperparameters: pd.Series,
    ):
        def matern_25(ard: bool, lengthscale_prior: AnyPrior) -> MaternKernel:
            return MaternKernel(nu=2.5, lengthscale_prior=lengthscale_prior, ard=ard)

        def matern_15(ard: bool, lengthscale_prior: AnyPrior) -> MaternKernel:
            return MaternKernel(nu=1.5, lengthscale_prior=lengthscale_prior, ard=ard)

        if hyperparameters.prior == "mbo":
            noise_prior, lengthscale_prior = (
                MBO_NOISE_PRIOR(),
                MBO_LENGTHSCALE_PRIOR(),
            )
        elif hyperparameters.prior == "threesix":
            noise_prior, lengthscale_prior = (
                THREESIX_NOISE_PRIOR(),
                THREESIX_LENGTHSCALE_PRIOR(),
            )
        else:
            noise_prior, lengthscale_prior = (
                HVARFNER_NOISE_PRIOR(),
                HVARFNER_LENGTHSCALE_PRIOR(),
            )

        surrogate_data.noise_prior = noise_prior
        if hyperparameters.kernel == "rbf":
            surrogate_data.kernel = RBFKernel(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
            )
        elif hyperparameters.kernel == "matern_2.5":
            surrogate_data.kernel = matern_25(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
            )
        elif hyperparameters.kernel == "matern_1.5":
            surrogate_data.kernel = matern_15(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
            )
        else:
            raise ValueError(f"Kernel {hyperparameters.kernel} not known.")


class MultiTaskGPSurrogate(TrainableBotorchSurrogate):
    """Gaussian process fitted jointly across several related tasks.

    Learns how strongly the tasks correlate, so observations of a cheap or abundant task
    inform the target one. Needs a `CategoricalTaskInput` naming the tasks.
    """

    type: Literal["MultiTaskGPSurrogate"] = "MultiTaskGPSurrogate"
    kernel: AnyKernel = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
        ),
        description=KERNEL_DESCRIPTION,
    )
    noise_prior: AnyPrior = Field(
        default_factory=lambda: HVARFNER_NOISE_PRIOR(),
        description=NOISE_PRIOR_DESCRIPTION,
    )
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default_factory=lambda: GreaterThan(lower_bound=1e-4),
        description=NOISE_CONSTRAINT_DESCRIPTION,
    )
    task_prior: Optional[LKJPrior] = Field(
        default=None,
        description="Prior over the correlations between tasks. If not provided, they "
        "are fitted without one.",
    )
    hyperconfig: Optional[MultiTaskGPHyperconfig] = Field(
        default_factory=lambda: MultiTaskGPHyperconfig(),
        description=HYPERCONFIG_DESCRIPTION,
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
