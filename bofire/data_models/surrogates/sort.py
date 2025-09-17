from typing import Literal, Optional, Type, Union

import pandas as pd
from pydantic import Field, model_validator

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import MaternKernel, RBFKernel, WassersteinKernel
from bofire.data_models.priors.api import (
    MBO_LENGTHSCALE_PRIOR,
    MBO_NOISE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    AnyPrior,
    LogNormalPrior,
)
from bofire.data_models.surrogates.trainable import Hyperconfig
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class SortingGPSurrogateHyperconfig(Hyperconfig):
    type: Literal["SortingLinearGPSurrogateHyperconfig"] = (
        "SortingeLinearGPSurrogateHyperconfig"
    )
    inputs: Inputs = Inputs(
        features=[
            CategoricalInput(
                key="continuous_kernel",
                categories=["rbf", "matern_1.5", "matern_2.5"],
            ),
            CategoricalInput(key="prior", categories=["mbo", "botorch"]),
            CategoricalInput(key="ard", categories=["True", "False"]),
        ],
    )
    target_metric: RegressionMetricsEnum = RegressionMetricsEnum.MAE
    hyperstrategy: Literal[
        "FractionalFactorialStrategy", "SoboStrategy", "RandomStrategy"
    ] = "FractionalFactorialStrategy"

    @staticmethod
    def _update_hyperparameters(
        surrogate_data: "SortingGPSurrogate",
        hyperparameters: pd.Series,
    ):
        if hyperparameters.prior == "mbo":
            noise_prior, lengthscale_prior, outputscale_prior = (
                MBO_NOISE_PRIOR(),
                MBO_LENGTHSCALE_PRIOR(),
                MBO_OUTPUTSCALE_PRIOR(),
            )
        else:
            noise_prior, lengthscale_prior, outputscale_prior = (
                THREESIX_NOISE_PRIOR(),
                THREESIX_LENGTHSCALE_PRIOR(),
                THREESIX_SCALE_PRIOR(),
            )
        surrogate_data.noise_prior = noise_prior
        surrogate_data.outputscale_prior = outputscale_prior

        if hyperparameters.continuous_kernel == "rbf":
            surrogate_data.continuous_kernel = RBFKernel(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
            )

        elif hyperparameters.continuous_kernel == "matern_2.5":
            surrogate_data.continuous_kernel = MaternKernel(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                nu=2.5,
            )

        elif hyperparameters.continuous_kernel == "matern_1.5":
            surrogate_data.continuous_kernel = MaternKernel(
                ard=hyperparameters.ard,
                lengthscale_prior=lengthscale_prior,
                nu=1.5,
            )

        else:
            raise ValueError(f"Kernel {hyperparameters.kernel} not known.")


class SortingGPSurrogate(TrainableBotorchSurrogate):
    """GP surrogate that is based on a `WassersteinKernel` for modeling functions
    that take a monotonically increasing piecewise linear function as input. The
    computation of the covariance between the piecewise linears is done by the
    Wasserstein distance kernel. The continuous features are modeled by a separate
    kernel, which can be either of Matern or RBF type. Both kernels are then combined
    by a product kernel.

    Attributes:
        x_keys: The keys of the features that are used as x values for the interpolation.
        y_keys: The keys of the features that are used as y values for the interpolation.
        continuous_keys: The keys of the features that are used for the continuous kernel.
        shape_kernel: The kernel to be used on the sorted features.
        continuous_kernel: The kernel that is used for the continuous features.
        outputscale_prior: Prior for the outputscale of the GP.
        noise_prior: Prior for the noise of the GP.
        hyperconfig: The hyperconfig that is used for training the GP.

    """

    type: Literal["SortingGPSurrogate"] = "SortingGPSurrogate"  # type: ignore
    x_keys: list[str]
    y_keys: list[str]
    continuous_keys: list[str]
    hyperconfig: Optional[SortingGPSurrogateHyperconfig] = Field(  # type: ignore
        default_factory=lambda: SortingGPSurrogateHyperconfig(),
    )

    shape_kernel: Union[WassersteinKernel, RBFKernel] = Field(
        default_factory=lambda: WassersteinKernel(
            squared=False,
            lengthscale_prior=LogNormalPrior(loc=1.0, scale=2.0),
        ),
    )

    continuous_kernel: Optional[Union[RBFKernel, MaternKernel]] = Field(
        default_factory=lambda: RBFKernel(
            lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR(),
        )
    )

    outputscale_prior: AnyPrior = Field(default_factory=lambda: THREESIX_SCALE_PRIOR())
    noise_prior: AnyPrior = Field(default_factory=lambda: THREESIX_NOISE_PRIOR())
    ard: bool = False

    @model_validator(mode="after")
    def validate_keys(self):
        if (
            sorted(set(self.x_keys + self.y_keys + self.continuous_keys))
            != self.inputs.get_keys()
        ):
            raise ValueError("Feature keys do not match input keys.")
        if len(self.x_keys) == 0 or len(self.y_keys) == 0:
            raise ValueError(
                "No features for sorting. Please provide `x_keys` and `y_keys`.",
            )
        return self

    @model_validator(mode="after")
    def validate_continuous_kernel(self):
        if len(self.continuous_keys) == 0 and self.continuous_kernel is not None:
            raise ValueError(
                "Continuous kernel specified but no features for continuous kernel.",
            )
        return self

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
