from typing import Literal, Optional

from pydantic import Field, PositiveFloat, PositiveInt

from bofire.data_models.base import BaseModel


class LBFGSB(BaseModel):
    """L-BFGS-B, the optimizer that fits a Gaussian process's hyperparameters by
    maximizing the marginal likelihood.

    Options left at None keep BoTorch's defaults: scipy's for an ordinary fit, and
    its own (at most 10000 iterations, tolerances of 1e-8) for the relevance pursuit
    of a GP with a `SparseOutlierGaussianLikelihood`.

    Examples:
        >>> LBFGSB(maxiter=200)
    """

    type: Literal["LBFGSB"] = "LBFGSB"
    maxiter: Optional[PositiveInt] = Field(
        default=None,
        description="Maximum number of iterations per fit attempt. If not provided, "
        "scipy's default of 15000.",
    )
    ftol: Optional[PositiveFloat] = Field(
        default=None,
        description="The fit stops once the relative change of the marginal "
        "likelihood falls below this value. If not provided, scipy's default of "
        "about 2.2e-9.",
    )
    gtol: Optional[PositiveFloat] = Field(
        default=None,
        description="The fit stops once the largest component of the projected "
        "gradient falls below this value. If not provided, scipy's default of 1e-5.",
    )
    max_attempts: PositiveInt = Field(
        default=10,
        description="Number of attempts in total. When an attempt fails or the "
        "optimizer warns, e.g. because it did not converge, the fit is retried from "
        "hyperparameters sampled from their priors. The first successful attempt is "
        "kept; if none succeeds, fitting raises an error. Not used by relevance "
        "pursuit.",
    )


class LBFGSBFitted(BaseModel):
    """A surrogate whose hyperparameters are fitted with L-BFGS-B."""

    optimizer: LBFGSB = Field(
        default=LBFGSB(),
        description="Optimizer that fits the hyperparameters, and its options.",
    )
