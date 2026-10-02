import math
from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
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


class GaussianLikelihood(Likelihood):
    """Observations are the response plus Gaussian noise with mean zero and one
    variance shared by all points.

    The noise variance is fitted along with the kernel hyperparameters;
    `noise_prior` regularises it and `noise_constraint` bounds it. The defaults put a
    log-normal prior on it that favours small noise levels and start the fit at that
    prior's mode, which suits outputs standardized to unit variance.
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


class PairwiseProbitLikelihood(Likelihood):
    """Binary preferences with Gaussian noise on the latent utilities.

    Converts the difference in utility between two alternatives into a preference
    probability using the cumulative normal distribution.
    """

    type: Literal["PairwiseProbitLikelihood"] = "PairwiseProbitLikelihood"


class PairwiseLogitLikelihood(Likelihood):
    """Binary preferences described by the Bradley-Terry model.

    Converts the difference in utility between two alternatives into a preference
    probability using the logistic sigmoid function.
    """

    type: Literal["PairwiseLogitLikelihood"] = "PairwiseLogitLikelihood"
