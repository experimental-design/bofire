from typing import Literal, Type

from pydantic import Field, PositiveInt, model_validator, validator

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import AnyKernel, ScaleKernel, WassersteinKernel
from bofire.data_models.priors.api import (
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
    AnyPrior,
    LogNormalPrior,
)
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate
from bofire.data_models.types import Bounds


class PiecewiseLinearGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["PiecewiseLinearGPSurrogate"] = "PiecewiseLinearGPSurrogate"
    interpolation_range: Bounds
    n_interpolation_points: PositiveInt = 400
    x_keys: list[str]
    y_keys: list[str]
    x_bounds: Bounds
    y_bounds: Bounds

    kernel: AnyKernel = Field(
        default_factory=lambda: ScaleKernel(
            base_kernel=WassersteinKernel(
                squared=False,
                lengthscale_prior=LogNormalPrior(loc=1.0, scale=2.0),
            ),
            outputscale_prior=BOTORCH_SCALE_PRIOR(),
        )
    )
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())

    @validator("kernel")
    @classmethod
    def validate_kernel(cls, v, values):
        if isinstance(v, ScaleKernel):
            if not isinstance(v.base_kernel, WassersteinKernel):
                raise ValueError(
                    "PiecewiseLinearGPSurrogate can only be used with a Wasserstein kernel."
                )
        else:
            if not isinstance(v, WassersteinKernel):
                raise ValueError(
                    "PiecewiseLinearGPSurrogate can only be used with a Wasserstein kernel."
                )
        return v

    @model_validator(mode="after")
    def validate_keys(self):
        if len(self.x_keys) != len(self.y_keys):
            raise ValueError("Different number of x and y keys.")
        if sorted(self.x_keys + self.y_keys) != self.inputs.get_keys():
            raise ValueError("Feature keys do not match input keys.")
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
