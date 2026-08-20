from typing import Literal

from pydantic import Field, PositiveFloat

from bofire.data_models.priors.prior import Prior


class GammaPrior(Prior):
    """Gamma prior based on the gamma distribution.

    Examples:
        >>> GammaPrior(concentration=3.0, rate=6.0)
    """

    type: Literal["GammaPrior"] = "GammaPrior"
    concentration: PositiveFloat = Field(
        description="Concentration (shape) of the gamma distribution.",
    )
    rate: PositiveFloat = Field(
        description="Rate (inverse scale) of the gamma distribution.",
    )


class DimensionalityScaledGammaPrior(Prior):
    """Gamma prior whose concentration and rate are scaled by the dimensionality of
    the problem, so that the lengthscale mode can grow with the problem dimensionality.

    The effective gamma distribution used at mapping time (given the dimensionality
    ``d``) is::

        concentration_eff = concentration + concentration_scaling * sqrt(d)
        rate_eff          = rate * d ** rate_power

    The asymmetric scaling (additive on the concentration, power on the rate) makes it
    possible to express both the CHEN priors (concentration growing with sqrt(d), rate
    fixed) and the dimensionality-scaled threesix prior (concentration fixed, rate
    decaying with sqrt(d)) with a single, serializable prior. See the constants in
    ``bofire.data_models.priors.api`` (``CHEN_*``,
    ``DIMENSIONALITY_SCALED_THREESIX_LENGTHSCALE_PRIOR``).

    Use instead of `GammaPrior` when the same prior should be reusable across problems
    of differing dimensionality.
    """

    type: Literal["DimensionalityScaledGammaPrior"] = "DimensionalityScaledGammaPrior"
    concentration: PositiveFloat = Field(
        default=3.0,
        description="Base concentration of the gamma distribution, before the "
        "dimensionality-dependent term is added.",
    )
    concentration_scaling: float = Field(
        default=0.0,
        description="Factor multiplying ``sqrt(d)`` that is added to the base "
        "concentration.",
    )
    rate: PositiveFloat = Field(
        default=6.0,
        description="Base rate of the gamma distribution, before the "
        "dimensionality-dependent scaling is applied.",
    )
    rate_power: float = Field(
        default=0.0,
        description="Exponent of ``d`` that the base rate is multiplied by.",
    )
