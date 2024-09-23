from typing import Annotated, List, Literal, Optional, Type, Union

from pydantic import AfterValidator, Field, PositiveInt, model_validator

# from bofire.data_models.strategies.api import FactorialStrategy
from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import MaternKernel, RBFKernel, WassersteinKernel
from bofire.data_models.priors.api import (
    BOTORCH_LENGTHCALE_PRIOR,
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
    AnyPrior,
    LogNormalPrior,
)
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate
from bofire.data_models.types import Bounds, validate_monotonically_increasing


class PiecewiseLinearGPSurrogate(TrainableBotorchSurrogate):
    type: Literal["PiecewiseLinearGPSurrogate"] = "PiecewiseLinearGPSurrogate"
    interpolation_range: Bounds
    n_interpolation_points: PositiveInt = 1000
    x_keys: list[str]
    y_keys: list[str]
    continuous_keys: list[str]
    prepend_x: Annotated[List[float], AfterValidator(validate_monotonically_increasing)]
    append_x: Annotated[List[float], AfterValidator(validate_monotonically_increasing)]
    prepend_y: Annotated[List[float], AfterValidator(validate_monotonically_increasing)]
    append_y: Annotated[List[float], AfterValidator(validate_monotonically_increasing)]

    shape_kernel: WassersteinKernel = Field(
        default_factory=lambda: WassersteinKernel(
            squared=False,
            lengthscale_prior=LogNormalPrior(loc=1.0, scale=2.0),
        )
    )

    continuous_kernel: Optional[Union[RBFKernel, MaternKernel]] = Field(
        default_factory=lambda: RBFKernel(
            lengthscale_prior=BOTORCH_LENGTHCALE_PRIOR(),
        )
    )

    outputscale_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_SCALE_PRIOR())
    noise_prior: AnyPrior = Field(default_factory=lambda: BOTORCH_NOISE_PRIOR())

    @model_validator(mode="after")
    def validate_keys(self):
        if (
            sorted(self.x_keys + self.y_keys + self.continuous_keys)
            != self.inputs.get_keys()
        ):
            raise ValueError("Feature keys do not match input keys.")
        if len(self.x_keys) == 0 or len(self.y_keys) == 0:
            raise ValueError(
                "No features for interpolation. Please provide `x_keys` and `y_keys`."
            )
        if len(self.x_keys) + len(self.append_x) + len(self.prepend_x) != len(
            self.y_keys
        ) + len(self.append_y) + len(self.prepend_y):
            raise ValueError("Different number of x and y values for interpolation.")
        return self

    @model_validator(mode="after")
    def validate_continuous_kernel(self):
        if len(self.continuous_keys) == 0 and self.continuous_kernel is not None:
            raise ValueError(
                "Continuous kernel specified but no features for continuous kernel."
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
