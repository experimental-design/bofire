from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.priors.api import AnyPrior
from bofire.data_models.types import StrictBounds


class Mean(BaseModel):
    """Prior mean function of a Gaussian process.

    The mean is what the model predicts far from any observation, where the data says
    nothing. The outputs are standardized before fitting by default, so a mean near zero
    corresponds to the average of the observations.
    """

    type: Any


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
    bounds: Optional[StrictBounds] = Field(
        default=None,
        description="Lower and upper bound the constant is restricted to during "
        "fitting. If not provided, it is unbounded.",
    )
