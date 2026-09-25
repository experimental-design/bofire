from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import Field, field_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Inputs
from bofire.data_models.encodings.api import AnyCategoricalEncoding, OrdinalEncoding
from bofire.data_models.feature_context import FeatureContext, task_input_key
from bofire.data_models.priors.api import AnyPrior


class Mean(BaseModel):
    """Prior mean function of a Gaussian process.

    The mean is what the model predicts far from any observation, where the data says
    nothing. The outputs are standardized before fitting by default, so a mean near zero
    corresponds to the average of the observations.
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


class ConstantMean(Mean):
    """Mean that is one constant everywhere, fitted to the data.

    With no prior and no bounds, the constant is fitted freely.
    """

    type: Literal["ConstantMean"] = "ConstantMean"
    prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the constant. If not provided, the constant is fitted "
        "without one.",
    )
    bounds: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Lower and upper bound the constant is restricted to during "
        "fitting. If not provided, it is unbounded.",
    )

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, bounds):
        if bounds is not None and bounds[0] >= bounds[1]:
            raise ValueError(
                f"The lower bound must be less than the upper bound, got {bounds}."
            )
        return bounds


class TaskConstantMean(ConstantMean):
    """Mean that is one constant per task, each fitted to that task's data.

    Tasks often differ by an offset -- a simulation that runs systematically high, an
    older campaign at a different baseline. A separate constant per task absorbs that
    offset, so the shared kernel only has to explain how the tasks co-vary. `prior`
    and `bounds` apply to each task's constant.
    """

    type: Literal["TaskConstantMean"] = "TaskConstantMean"
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
