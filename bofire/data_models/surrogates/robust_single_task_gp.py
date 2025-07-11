from typing import Literal, Optional, Type, Union

from pydantic import Field, model_validator

from bofire.data_models.features.api import AnyOutput, ContinuousOutput
from bofire.data_models.kernels.api import MaternKernel, RBFKernel, ScaleKernel
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    ROBUSTGP_LENGTHSCALE_CONSTRAINT,
    ROBUSTGP_OUTPUTSCALE_CONSTRAINT,
    AnyPrior,
)
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPHyperconfig
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class RobustSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """
    Robust Relevance Pursuit Single Task Gaussian Process Surrogate.

    A robust single-task GP that learns a data-point specific noise level and is therefore more robust to outliers.
    See: https://botorch.org/docs/tutorials/relevance_pursuit_robust_regression/
    Paper: https://arxiv.org/pdf/2410.24222

    Attributes:
        prior_mean_of_support: The prior mean of the support.
        convex_parametrization: Whether to use convex parametrization of the sparse noise model.
        cache_model_trace: Whether to cache the model trace. This needs no be set to True if you want to view the model trace after optimization.

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
    hyperconfig: Optional[SingleTaskGPHyperconfig] = Field(
        default_factory=lambda: SingleTaskGPHyperconfig(
            lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            outputscale_constraint=ROBUSTGP_OUTPUTSCALE_CONSTRAINT(),
        ),
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

    @model_validator(mode="after")
    def validate_number_of_outputs(self):
        if len(self.outputs.features) > 1:
            raise ValueError("RobustGP only supports one output.")
        return self
