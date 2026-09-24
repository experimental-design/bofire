from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.feature_context import FeatureContext
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class Mean(BaseModel):
    """Prior mean function of a Gaussian process.

    The mean is what the model predicts far from any observation, where the data says
    nothing. The outputs are standardized before fitting by default, so a mean near zero
    corresponds to the average of the observations.
    """

    type: Any

    def validate_inputs(self, context: FeatureContext) -> None:
        """Check that this component can work on what it is applied to.

        Selects no features, so there is nothing to check.

        Args:
            context: The features on offer and how they are encoded.
        """


class ConstantMean(Mean):
    """Mean that is one constant everywhere, fitted to the data.

    With no prior and no constraint, the constant is fitted freely.
    """

    type: Literal["ConstantMean"] = "ConstantMean"
    prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the constant. If not provided, the constant is fitted "
        "without one.",
    )
    constraint: Optional[AnyPriorConstraint] = Field(
        default=None,
        description="Bounds the constant is restricted to during fitting. If not "
        "provided, it is unbounded.",
    )
