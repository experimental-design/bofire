import math
from typing import Any, Dict, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs
from bofire.data_models.encodings.api import AnyCategoricalEncoding, OrdinalEncoding
from bofire.data_models.feature_context import FeatureContext, task_input_key
from bofire.data_models.priors.api import (
    HVARFNER_NOISE_PRIOR,
    AnyPrior,
    AnyPriorConstraint,
    GreaterThan,
)


class Likelihood(BaseModel):
    """Model of how observations scatter around the underlying response.

    It decides how much of the spread in the data is attributed to measurement noise
    rather than to the response itself.
    """

    type: Any

    def encoding_requests(self, inputs: Inputs) -> Dict[str, AnyCategoricalEncoding]:
        """The encodings this component needs for features left without one.

        Args:
            inputs: The inputs of the surrogate the component belongs to.

        Returns:
            Encodings by feature key; empty if the component has no need.
        """
        return {}

    def validate_inputs(self, context: FeatureContext) -> None:
        """Check that this component can work on what it is applied to.

        Selects no features, so there is nothing to check.

        Args:
            context: The features on offer and how they are encoded.
        """


class GaussianLikelihood(Likelihood):
    """Observations scatter around the response with Gaussian noise of one variance.

    The defaults put a log-normal prior on the noise that favours small noise levels
    and start the fit at that prior's mode, which suits outputs standardized to unit
    variance.
    """

    type: Literal["GaussianLikelihood"] = "GaussianLikelihood"
    noise_prior: AnyPrior = Field(
        default=HVARFNER_NOISE_PRIOR(),
        description="Prior over the noise variance, which sets how much of the spread "
        "in the data is attributed to measurement error rather than to the response.",
    )
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default=GreaterThan(lower_bound=1e-4, initial_value=math.exp(-5.0)),
        description="Bounds the noise variance is restricted to during fitting. A "
        "positive lower bound keeps the fit numerically stable.",
    )


class TaskGaussianLikelihood(GaussianLikelihood):
    """Gaussian noise with a separate variance for each task.

    Tasks are often measured with different precision -- a simulation next to a real
    experiment, a quick screen next to a careful measurement. With one variance per task
    the model trusts each task's observations according to its own noise level.
    `noise_prior` and `noise_constraint` apply to each task's variance.
    """

    type: Literal["TaskGaussianLikelihood"] = "TaskGaussianLikelihood"
    task_feature: Optional[str] = Field(
        default=None,
        description="Key of the task input. If not provided, the single task input "
        "of the domain.",
    )

    def encoding_requests(self, inputs: Inputs) -> Dict[str, AnyCategoricalEncoding]:
        key = task_input_key(inputs, self.task_feature)
        return {} if key is None else {key: OrdinalEncoding()}

    def validate_inputs(self, context: FeatureContext) -> None:
        """Check that there is a usable task input.

        Raises:
            ValueError: If there is no task input encoded as ordinal codes.
        """
        context.task_feature(self.task_feature)
