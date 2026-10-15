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


class SparseOutlierGaussianLikelihood(GaussianLikelihood):
    """Gaussian noise plus a noise level of its own for every observation, of which
    as few as possible are non-zero.

    Observations the model cannot explain get a large noise level of their own and
    are treated as outliers instead of distorting the fit; the number of outliers is
    chosen by Bayesian model comparison (relevance pursuit, Ament et al. 2024,
    https://arxiv.org/abs/2410.24222). `noise_prior` and `noise_constraint` refer to
    the noise shared by all observations. Works with one output only.

    What counts as an outlier depends on how flexible the model is: a kernel with
    short lengthscales can explain almost any point. Bound the lengthscales from
    below, e.g. with `ROBUSTGP_LENGTHSCALE_CONSTRAINT()`.
    """

    type: Literal["SparseOutlierGaussianLikelihood"] = "SparseOutlierGaussianLikelihood"
    prior_mean_of_support: Optional[int] = Field(
        default=None,
        description="Number of outliers expected a priori, the mean of the prior over "
        "the number of outliers used to compare the candidate models. If not "
        "provided, 20% of the observations.",
    )
    convex_parametrization: bool = Field(
        default=True,
        description="Whether the outlier noise levels are fitted in a convex "
        "parametrization, which generally gives better fits.",
    )
    cache_model_trace: bool = Field(
        default=False,
        description="Whether to keep every model fitted along the way, one per number "
        "of outliers, on the fitted model for inspection.",
    )
