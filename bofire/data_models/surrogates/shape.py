from typing import Annotated, List, Literal, Optional, Type, Union

import pandas as pd
from pydantic import AfterValidator, Field, PositiveFloat, PositiveInt, model_validator

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import MaternKernel, RBFKernel, WassersteinKernel
from bofire.data_models.priors.api import (
    MBO_LENGTHCALE_PRIOR,
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
from bofire.data_models.types import Bounds, validate_monotonically_increasing


class PiecewiseLinearGPSurrogateHyperconfig(Hyperconfig):
    type: Literal["PiecewiseLinearGPSurrogateHyperconfig"] = (
        "PiecewiseLinearGPSurrogateHyperconfig"
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
        surrogate_data: "PiecewiseLinearGPSurrogate",
        hyperparameters: pd.Series,
    ):
        if hyperparameters.prior == "mbo":
            noise_prior, lengthscale_prior, outputscale_prior = (
                MBO_NOISE_PRIOR(),
                MBO_LENGTHCALE_PRIOR(),
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


class PiecewiseLinearGPSurrogate(TrainableBotorchSurrogate):
    """GP surrogate that is based on a `WassersteinKernel` for modeling functions
    that take a monotonically increasing piecewise linear function as input. The
    computation of the covariance between the piecewise linears is done by the
    Wasserstein distance kernel. The continuous features are modeled by a separate
    kernel, which can be either of Matern or RBF type. Both kernels are then combined
    by a product kernel.

    Attributes:
        interpolation_range: The range of the interpolation for the piecewise linear
            functions.
        n_interpolation_points: Number of interpolation points in the interpolation range.
        x_keys: The keys of the features that are used as x values for the interpolation.
        y_keys: The keys of the features that are used as y values for the interpolation.
        continuous_keys: The keys of the features that are used for the continuous kernel.
        prepend_x: The x values that are prepended to the values of the `x_keys`.
        append_x: The x values that are appended to the values of the `x_keys`.
        prepend_y: The y values that are prepended to the values of the `y_keys`.
        append_y: The y values that are appended to the values of the `y_keys`.
        shape_kernel: The Wasserstein distance kernel to be used.
        continuous_kernel: The kernel that is used for the continuous features.
        outputscale_prior: Prior for the outputscale of the GP.
        noise_prior: Prior for the noise of the GP.
        hyperconfig: The hyperconfig that is used for training the GP.

    """

    type: Literal["PiecewiseLinearGPSurrogate"] = "PiecewiseLinearGPSurrogate"  # type: ignore
    interpolation_range: Bounds
    n_interpolation_points: PositiveInt = 1000
    x_keys: list[str]
    y_keys: list[str]
    continuous_keys: list[str]
    prepend_x: Annotated[List[float], AfterValidator(validate_monotonically_increasing)]
    append_x: Annotated[List[float], AfterValidator(validate_monotonically_increasing)]
    prepend_y: Annotated[List[float], AfterValidator(validate_monotonically_increasing)]
    append_y: Annotated[List[float], AfterValidator(validate_monotonically_increasing)]
    normalize_y: PositiveFloat = 1.0
    hyperconfig: Optional[PiecewiseLinearGPSurrogateHyperconfig] = Field(  # type: ignore
        default_factory=lambda: PiecewiseLinearGPSurrogateHyperconfig(),
    )

    shape_kernel: WassersteinKernel = Field(
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

    @model_validator(mode="after")
    def validate_keys(self):
        if (
            sorted(set(self.x_keys + self.y_keys + self.continuous_keys))
            != self.inputs.get_keys()
        ):
            raise ValueError("Feature keys do not match input keys.")
        if len(self.x_keys) == 0 or len(self.y_keys) == 0:
            raise ValueError(
                "No features for interpolation. Please provide `x_keys` and `y_keys`.",
            )
        if len(self.x_keys) + len(self.append_x) + len(self.prepend_x) != len(
            self.y_keys,
        ) + len(self.append_y) + len(self.prepend_y):
            raise ValueError("Different number of x and y values for interpolation.")
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
