from typing import Literal, Optional, Type, Union

import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import MaternKernel, RBFKernel, ScaleKernel
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    MBO_LENGTHCALE_PRIOR,
    MBO_NOISE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
    ROBUSTGP_LENGTHSCALE_CONSTRAINT,
    ROBUSTGP_OUTPUTSCALE_CONSTRAINT,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    AnyConstraint,
    AnyPrior,
)
from bofire.data_models.surrogates.trainable import Hyperconfig
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class RobustSingleTaskGPHyperconfig(Hyperconfig):
    type: Literal["RobustSingleTaskGPHyperconfig"] = "RobustSingleTaskGPHyperconfig"
    inputs: Inputs = Inputs(
        features=[
            CategoricalInput(
                key="kernel",
                categories=["rbf", "matern_1.5", "matern_2.5"],
            ),
            CategoricalInput(key="prior", categories=["mbo", "threesix", "hvarfner"]),
            CategoricalInput(key="scalekernel", categories=["True", "False"]),
            CategoricalInput(key="ard", categories=["True", "False"]),
        ],
    )
    target_metric: RegressionMetricsEnum = RegressionMetricsEnum.MAE
    hyperstrategy: Literal[
        "FractionalFactorialStrategy", "SoboStrategy", "RandomStrategy"
    ] = "FractionalFactorialStrategy"

    @staticmethod
    def _update_hyperparameters(
        surrogate_data: "RobustSingleTaskGPSurrogate",
        hyperparameters: pd.Series,
    ):
        def matern_25(
            ard: bool,
            lengthscale_prior: AnyPrior,
            lengthscale_constraint: AnyConstraint,
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
            lengthscale_constraint: AnyConstraint,
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
                MBO_LENGTHCALE_PRIOR(),
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
                lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            )
        elif hyperparameters.kernel == "matern_2.5":
            base_kernel = matern_25(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            )
        elif hyperparameters.kernel == "matern_1.5":
            base_kernel = matern_15(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            )
        else:
            raise ValueError(f"Kernel {hyperparameters.kernel} not known.")

        if hyperparameters.scalekernel:
            surrogate_data.kernel = ScaleKernel(
                base_kernel=base_kernel,
                outputscale_prior=outputscale_prior,
                outputscale_constraint=ROBUSTGP_OUTPUTSCALE_CONSTRAINT(),
            )
        else:
            surrogate_data.kernel = base_kernel


class RobustSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """
    Robust Relevance Pursuit Single Task Gaussian Process Surrogate.

    A robust single-task GP that learns a data-point specific noise level and is therefore more robust to outliers.
    See: https://botorch.org/docs/tutorials/relevance_pursuit_robust_regression/
    Paper: https://arxiv.org/pdf/2410.24222

    Attributes:
        prior_mean_of_support (float): The prior mean of the support.
        convex_parametrization (bool): Whether to use convex parametrization of the sparse noise model.
        cache_model_trace (bool): Whether to cache the model trace. This needs no be set to True if you want to view the model trace after optimization.

    Note:
        The definition of "outliers" depends on the model capacity, so what is an outlier
        with respect to a simple model might not be an outlier with respect to a complex model.
        For this reason, it is necessary to bound the lengthscale of the GP kernel from below.
    """

    type: Literal["RobustSingleTaskGPSurrogate"] = "RobustSingleTaskGPSurrogate"

    kernel: Union[ScaleKernel, RBFKernel, MaternKernel] = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
            lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: HVARFNER_NOISE_PRIOR())
    hyperconfig: Optional[RobustSingleTaskGPHyperconfig] = Field(
        default_factory=lambda: RobustSingleTaskGPHyperconfig(),
    )

    prior_mean_of_support: Optional[int] = Field(default=None)
    convex_parametrization: bool = Field(default=True)
    cache_model_trace: bool = Field(default=False)

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))

    # check that there is only one output
    @field_validator("num outputs", check_fields=False)
    @classmethod
    def validate_outputs(cls, outputs):
        if len(outputs) > 1:
            raise ValueError("RobustGP only supports one output.")
