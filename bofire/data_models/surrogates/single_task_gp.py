from typing import Literal, Optional, Type, Union

import pandas as pd
from pydantic import Field

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import (
    AnyKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    MBO_LENGTHSCALE_PRIOR,
    MBO_NOISE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)
from bofire.data_models.surrogates.trainable import (
    HYPERCONFIG_INPUTS_DESCRIPTION,
    HYPERSTRATEGY_DESCRIPTION,
    TARGET_METRIC_DESCRIPTION,
    Hyperconfig,
)
from bofire.data_models.surrogates.trainable_botorch import (
    HYPERCONFIG_DESCRIPTION,
    NOISE_CONSTRAINT_DESCRIPTION,
    NOISE_PRIOR_DESCRIPTION,
    TrainableBotorchSurrogate,
)


class SingleTaskGPHyperconfig(Hyperconfig):
    """Hyperparameter search for a single-task GP: kernel, prior family, scaling, ARD."""

    type: Literal["SingleTaskGPHyperconfig"] = "SingleTaskGPHyperconfig"
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
                CategoricalInput(key="scalekernel", categories=["True", "False"]),
                CategoricalInput(key="ard", categories=["True", "False"]),
            ],
        ),
        description=HYPERCONFIG_INPUTS_DESCRIPTION,
    )
    lengthscale_constraint: Optional[AnyPriorConstraint] = Field(
        default=None,
        description="Bounds applied to the lengthscale of whichever kernel the search "
        "picks, so that the searched kernels share one restriction.",
    )
    outputscale_constraint: Optional[AnyPriorConstraint] = Field(
        default=None,
        description="Bounds applied to the output scale when the search selects a "
        "scaled kernel.",
    )
    target_metric: RegressionMetricsEnum = Field(
        default=RegressionMetricsEnum.MAE, description=TARGET_METRIC_DESCRIPTION
    )
    hyperstrategy: Literal[
        "FractionalFactorialStrategy", "SoboStrategy", "RandomStrategy"
    ] = Field(
        default="FractionalFactorialStrategy", description=HYPERSTRATEGY_DESCRIPTION
    )

    @staticmethod
    def _update_hyperparameters(
        surrogate_data: "SingleTaskGPSurrogate",
        hyperparameters: pd.Series,
        outputscale_constraint: Optional[AnyPriorConstraint] = None,
        lengthscale_constraint: Optional[AnyPriorConstraint] = None,
    ):
        def matern_25(
            ard: bool,
            lengthscale_prior: AnyPrior,
            lengthscale_constraint: Union[AnyPriorConstraint, None],
        ) -> MaternKernel:
            return MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
                ard=ard,
            )

        def matern_15(
            ard: bool,
            lengthscale_prior: AnyPrior,
            lengthscale_constraint: Union[AnyPriorConstraint, None],
        ) -> MaternKernel:
            return MaternKernel(
                nu=1.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
                ard=ard,
            )

        if hyperparameters.prior == "mbo":
            noise_prior, lengthscale_prior, outputscale_prior = (
                MBO_NOISE_PRIOR(),
                MBO_LENGTHSCALE_PRIOR(),
                MBO_OUTPUTSCALE_PRIOR(),
            )
        elif hyperparameters.prior == "threesix":
            noise_prior, lengthscale_prior, outputscale_prior = (
                THREESIX_NOISE_PRIOR(),
                THREESIX_LENGTHSCALE_PRIOR(),
                THREESIX_SCALE_PRIOR(),
            )
        else:
            noise_prior, lengthscale_prior, outputscale_prior = (
                HVARFNER_NOISE_PRIOR(),
                HVARFNER_LENGTHSCALE_PRIOR(),
                THREESIX_SCALE_PRIOR(),
            )
        surrogate_data.noise_prior = noise_prior

        if hyperparameters.kernel == "rbf":
            base_kernel = RBFKernel(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
            )
        elif hyperparameters.kernel == "matern_2.5":
            base_kernel = matern_25(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
            )
        elif hyperparameters.kernel == "matern_1.5":
            base_kernel = matern_15(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=lengthscale_constraint,
            )
        else:
            raise ValueError(f"Kernel {hyperparameters.kernel} not known.")

        if hyperparameters.scalekernel == "True":
            surrogate_data.kernel = ScaleKernel(
                base_kernel=base_kernel,
                outputscale_prior=outputscale_prior,
                outputscale_constraint=outputscale_constraint,
            )
        else:
            surrogate_data.kernel = base_kernel


class SingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Gaussian process over one output, the usual starting point.

    Predicts a mean and an uncertainty everywhere, which is what lets a Bayesian
    optimization strategy trade exploration against exploitation.

    Examples:
        >>> surrogate = SingleTaskGPSurrogate(inputs=inputs, outputs=outputs)

        A rougher response than the default RBF assumes:

        >>> surrogate = SingleTaskGPSurrogate(
        ...     inputs=inputs, outputs=outputs, kernel=MaternKernel(nu=1.5, ard=True)
        ... )
    """

    type: Literal["SingleTaskGPSurrogate"] = "SingleTaskGPSurrogate"

    kernel: AnyKernel = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
        ),
        description="Covariance function, encoding what the model assumes about the "
        "response: how smooth it is and which inputs matter.",
    )
    noise_prior: AnyPrior = Field(
        default_factory=lambda: HVARFNER_NOISE_PRIOR(),
        description=NOISE_PRIOR_DESCRIPTION,
    )
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default_factory=lambda: GreaterThan(lower_bound=1e-4),
        description=NOISE_CONSTRAINT_DESCRIPTION,
    )
    hyperconfig: Optional[SingleTaskGPHyperconfig] = Field(
        default_factory=lambda: SingleTaskGPHyperconfig(),
        description=HYPERCONFIG_DESCRIPTION,
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
