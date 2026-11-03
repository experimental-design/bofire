from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint


class Mean(BaseModel):
    """Prior mean function of a Gaussian process.

    The mean is what the model predicts far from any observation, where the data says
    nothing. The outputs are standardized before fitting by default, so a mean near zero
    corresponds to the average of the observations.
    """

    type: Any


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
        description="Bounds the constant is restricted to during fitting, e.g. "
        "NonTransformedInterval(lower_bound=-10, upper_bound=10). If not provided, it "
        "is unbounded.",
    )


class TaskwiseMean(Mean):
    """One mean per task of a multi-task Gaussian process, so tasks may differ in
    offset.

    Every task gets a copy of `mean` with parameters of its own, each with its prior
    and bounds. A plain mean instead is shared by all tasks.
    """

    type: Literal["TaskwiseMean"] = "TaskwiseMean"
    mean: ConstantMean = Field(
        default=ConstantMean(),
        description="The mean every task gets a copy of; each copy is fitted "
        "separately.",
    )
