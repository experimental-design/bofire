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
    AnyPriorConstraint,
    GreaterThan,
)
from bofire.data_models.surrogates.single_task_gp import SingleTaskGPHyperconfig
from bofire.data_models.surrogates.trainable_botorch import (
    HYPERCONFIG_DESCRIPTION,
    NOISE_CONSTRAINT_DESCRIPTION,
    NOISE_PRIOR_DESCRIPTION,
    TrainableBotorchSurrogate,
)


class RobustSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    """Gaussian process that learns which experiments to distrust.

    Rather than one noise level for all data, it fits a per-point one and lets a few
    points take a large value, so a failed or mistyped experiment is discounted instead
    of dragging the fit towards it. Pick it over `SingleTaskGPSurrogate` when the data
    is expected to contain outliers that cannot be identified up front.

    Note:
        What counts as an outlier depends on how flexible the model is: a wiggly enough
        model explains any point. The lengthscale is therefore bounded from below, and
        loosening that bound weakens the robustness.

    References:
        Ament et al., Robust Gaussian Processes via Relevance Pursuit (2024),
        https://arxiv.org/abs/2410.24222
    """

    type: Literal["RobustSingleTaskGPSurrogate"] = "RobustSingleTaskGPSurrogate"

    kernel: Union[ScaleKernel, RBFKernel, MaternKernel] = Field(
        default_factory=lambda: RBFKernel(
            ard=True,
            lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
            lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
        ),
        description="Covariance function. Restricted to the stationary kernels, and "
        "its lengthscale is bounded from below, because what counts as an outlier "
        "depends on how flexible the model is allowed to be.",
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
        default_factory=lambda: SingleTaskGPHyperconfig(
            lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            outputscale_constraint=ROBUSTGP_OUTPUTSCALE_CONSTRAINT(),
        ),
        description=HYPERCONFIG_DESCRIPTION,
    )

    prior_mean_of_support: Optional[int] = Field(
        default=None,
        description="Expected number of points treated as outliers. If not provided, "
        "it is inferred during fitting.",
    )
    convex_parametrization: bool = Field(
        default=True,
        description="Whether to parametrize the sparse noise model convexly, which "
        "makes the fit better behaved.",
    )
    cache_model_trace: bool = Field(
        default=False,
        description="Whether to keep the sequence of models explored during fitting, "
        "which is needed to inspect the trace afterwards and costs memory.",
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

    @model_validator(mode="after")
    def validate_number_of_outputs(self):
        if len(self.outputs.features) > 1:
            raise ValueError("RobustGP only supports one output.")
        return self
