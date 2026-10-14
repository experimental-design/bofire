from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint, GreaterThan


class Likelihood(BaseModel):
    """Model of how observations scatter around the underlying response.

    It decides how much of the spread in the data is attributed to measurement noise
    rather than to the response itself.
    """

    type: Any


class GaussianLikelihood(Likelihood):
    """Observations are the response plus Gaussian noise with mean zero and one
    variance shared by all points.

    The noise variance is fitted along with the kernel hyperparameters;
    `noise_prior` regularises it and `noise_constraint` bounds it. By default there is
    no prior and the noise is at least 1e-4.

    The noise is that of the outputs as the model sees them, after the surrogate's
    output scaler (standardized by default), so priors and bounds refer to that scale
    and not to the raw measurements.
    """

    type: Literal["GaussianLikelihood"] = "GaussianLikelihood"
    noise_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the noise variance, which sets how much of the spread "
        "in the data is attributed to measurement error rather than to the response. "
        "If not provided, the noise is fitted without a prior.",
    )
    noise_constraint: Optional[AnyPriorConstraint] = Field(
        default=GreaterThan(lower_bound=1e-4),
        description="Bounds the noise variance is restricted to during fitting. A "
        "positive lower bound keeps the fit numerically stable.",
    )
