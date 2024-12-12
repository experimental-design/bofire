from typing import Literal, Optional, Type

import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum, RegressionMetricsEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousOutput,
    TaskInput,
)
from bofire.data_models.kernels.api import AnyKernel, MaternKernel, RBFKernel
from bofire.data_models.priors.api import (
    MBO_LENGTHCALE_PRIOR,
    MBO_NOISE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    AnyPrior,
)
from bofire.data_models.priors.lkj import LKJPrior
from bofire.data_models.surrogates.trainable import Hyperconfig
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class MultiTaskGPHyperconfig(Hyperconfig):
    type: Literal["MultiTaskGPHyperconfig"] = "MultiTaskGPHyperconfig"
    inputs: Inputs = Inputs(
        features=[
            CategoricalInput(
                key="kernel",
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
        surrogate_data: "MultiTaskGPSurrogate",
        hyperparameters: pd.Series,
    ):
        def matern_25(ard: bool, lengthscale_prior: AnyPrior) -> MaternKernel:
            return MaternKernel(nu=2.5, lengthscale_prior=lengthscale_prior, ard=ard)

        def matern_15(ard: bool, lengthscale_prior: AnyPrior) -> MaternKernel:
            return MaternKernel(nu=1.5, lengthscale_prior=lengthscale_prior, ard=ard)

        if hyperparameters.prior == "mbo":
            noise_prior, lengthscale_prior = (MBO_NOISE_PRIOR(), MBO_LENGTHCALE_PRIOR())
        else:
            noise_prior, lengthscale_prior = (
                THREESIX_NOISE_PRIOR(),
                THREESIX_LENGTHSCALE_PRIOR(),
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
    type: Literal["MultiTaskGPSurrogate"] = "MultiTaskGPSurrogate"
    kernel: AnyKernel = Field(
        default_factory=lambda: MaternKernel(
            ard=True,
            nu=2.5,
            lengthscale_prior=THREESIX_LENGTHSCALE_PRIOR(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: THREESIX_NOISE_PRIOR())
    task_prior: Optional[LKJPrior] = Field(default_factory=lambda: None)
    hyperconfig: Optional[MultiTaskGPHyperconfig] = Field(
        default_factory=lambda: MultiTaskGPHyperconfig(),
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

    @field_validator("inputs")
    @classmethod
    def validate_task_inputs(cls, inputs: Inputs):
        if len(inputs.get_keys(TaskInput)) != 1:
            raise ValueError("Exactly one task input is required for multi-task GPs.")
        return inputs

    @field_validator("input_preprocessing_specs")
    @classmethod
    def validate_encoding(cls, v, info):
        # also validate that the task feature has ordinal encoding
        if "inputs" not in info.data:
            return v

        if len(info.data["inputs"].get_keys(TaskInput)) == 0:
            return v

        task_feature_id = info.data["inputs"].get_keys(TaskInput)[0]
        if v.get(task_feature_id) is None:
            v[task_feature_id] = CategoricalEncodingEnum.ORDINAL
        elif v[task_feature_id] != CategoricalEncodingEnum.ORDINAL:
            raise ValueError(
                f"The task feature {task_feature_id} has to be encoded as ordinal.",
            )

        return v
