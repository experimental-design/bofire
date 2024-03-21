from typing import Literal, Optional, Type

import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum, RegressionMetricsEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import (
    AnyCategoricalKernel,
    AnyContinuousKernel,
    HammingDistanceKernel,
    MaternKernel,
    RBFKernel,
)
from bofire.data_models.priors.api import (
    BOTORCH_LENGTHCALE_PRIOR,
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
    MBO_LENGTHCALE_PRIOR,
    MBO_NOISE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
    AnyPrior,
)

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.surrogates.trainable import Hyperconfig
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MixedSingleTaskGPHyperconfig(Hyperconfig):
    type: Literal["MixedSingleTaskGPHyperconfig"] = "MixedSingleTaskGPHyperconfig"
    inputs: Inputs = Inputs(
        features=[
            CategoricalInput(
                key="continuous_kernel", categories=["rbf", "matern_1.5", "matern_2.5"]
            ),
            CategoricalInput(key="prior", categories=["mbo", "botorch"]),
            CategoricalInput(key="ard", categories=["True", "False"]),
        ]
    )
    target_metric: RegressionMetricsEnum = RegressionMetricsEnum.MAE
    hyperstrategy: Literal[
        "FactorialStrategy", "SoboStrategy", "RandomStrategy"
    ] = "FactorialStrategy"

    @staticmethod
    def _update_hyperparameters(
        surrogate_data: "MixedSingleTaskGPSurrogate", hyperparameters: pd.Series
    ):
        if hyperparameters.prior == "mbo":
            noise_prior, lengthscale_prior, _ = (
                MBO_NOISE_PRIOR(),
                MBO_LENGTHCALE_PRIOR(),
                MBO_OUTPUTSCALE_PRIOR(),
            )
        else:
            noise_prior, lengthscale_prior, _ = (
                BOTORCH_NOISE_PRIOR(),
                BOTORCH_LENGTHCALE_PRIOR(),
                BOTORCH_SCALE_PRIOR(),
            )
        surrogate_data.noise_prior = noise_prior
        if hyperparameters.continuous_kernel == "rbf":
            surrogate_data.continuous_kernel = RBFKernel(
                ard=hyperparameters.ard, lengthscale_prior=lengthscale_prior
            )

        elif hyperparameters.continuous_kernel == "matern_2.5":
            surrogate_data.continuous_kernel = MaternKernel(
                ard=hyperparameters.ard, lengthscale_prior=lengthscale_prior, nu=2.5
            )

        elif hyperparameters.continuous_kernel == "matern_1.5":
            surrogate_data.continuous_kernel = MaternKernel(
                ard=hyperparameters.ard, lengthscale_prior=lengthscale_prior, nu=1.5
            )

        else:
            raise ValueError(f"Kernel {hyperparameters.kernel} not known.")


class MixedSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["MixedSingleTaskGPSurrogate"] = "MixedSingleTaskGPSurrogate"
    continuous_kernel: AnyContinuousKernel = Field(
        default_factory=lambda: MaternKernel(
            ard=True, nu=2.5, lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR()
        )
    )
    categorical_kernel: AnyCategoricalKernel = Field(
        default_factory=lambda: HammingDistanceKernel(ard=True)
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())
    hyperconfig: Optional[MixedSingleTaskGPHyperconfig] = Field(
        default_factory=lambda: MixedSingleTaskGPHyperconfig()
    )

    @field_validator("input_preprocessing_specs")
    @classmethod
    def validate_categoricals(cls, v, values):
        """Checks that at least one one-hot encoded categorical feauture is present."""
        if CategoricalEncodingEnum.ONE_HOT not in v.values():
            raise ValueError(
                "MixedSingleTaskGPSurrogate can only be used if at least one one-hot encoded categorical feature is present."
            )
        return v

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Abstract method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
