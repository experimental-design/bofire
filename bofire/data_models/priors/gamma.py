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
