from typing import Literal

from pydantic import PositiveFloat

from bofire.data_models.priors.prior import Prior


class GammaPrior(Prior):
    """Gamma prior based on the gamma distribution

    Attributes:
        concentration(PostiveFloat): concentration of the gamma distribution
        rate(PositiveFloat): rate of the gamma prior.

    """

    type: Literal["GammaPrior"] = "GammaPrior"
    concentration: PositiveFloat
    rate: PositiveFloat


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

    Attributes:
        concentration(PositiveFloat): base concentration of the gamma distribution.
        concentration_scaling(float): factor multiplying ``sqrt(d)`` that is added to
            the base concentration.
        rate(PositiveFloat): base rate of the gamma distribution.
        rate_power(float): exponent of ``d`` that the base rate is multiplied by.

    """

    type: Literal["DimensionalityScaledGammaPrior"] = "DimensionalityScaledGammaPrior"
    concentration: PositiveFloat = 3.0
    concentration_scaling: float = 0.0
    rate: PositiveFloat = 6.0
    rate_power: float = 0.0
