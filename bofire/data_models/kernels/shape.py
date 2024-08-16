from typing import Literal, Optional

from bofire.data_models.kernels.kernel import Kernel
from bofire.data_models.priors.api import AnyPrior


class WassersteinKernel(Kernel):
    type: Literal["WassersteinKernel"] = "WassersteinKernel"
    squared: bool = False
    lengthscale_prior: Optional[AnyPrior] = None
