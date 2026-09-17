from enum import Enum
from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.types import NonRestrictedFeatureKeys
from bofire.data_models.unions import tagged_union


class ScalerEnum(str, Enum):
    """Rescalings available for a surrogate's output.

    `IDENTITY` leaves the output alone, `STANDARDIZE` centres it and scales it to unit
    variance, `LOG` takes its logarithm, and `CHAINED_LOG_STANDARDIZE` does both in that
    order, which suits a strictly positive output spanning orders of magnitude.
    """

    STANDARDIZE = "STANDARDIZE"
    IDENTITY = "IDENTITY"
    LOG = "LOG"
    CHAINED_LOG_STANDARDIZE = "CHAINED_LOG_STANDARDIZE"


class Scaler(BaseModel):
    """Rescaling applied to the inputs before the surrogate sees them.

    Kernels compare inputs by distance, so features on very different numeric ranges
    would otherwise contribute unequally regardless of how much they matter.
    """

    type: Any
    features: NonRestrictedFeatureKeys = Field(
        default=[],
        description="Keys of the input or engineered features to rescale. Empty means "
        "every feature that is still numeric after encoding, which leaves the columns "
        "an encoded categorical produces untouched.",
    )


class Normalize(Scaler):
    """Rescales each feature onto the unit interval, using the bounds declared on it.

    An engineered feature has no declared bounds, so its range is taken from the
    experiments instead.
    """

    type: Literal["Normalize"] = "Normalize"


class Standardize(Scaler):
    """Rescales each feature to zero mean and unit variance, using the observed data.

    Unlike `Normalize`, which uses the declared bounds, this follows the experiments, so
    it suits a feature whose bounds are wide relative to the region actually explored.
    """

    type: Literal["Standardize"] = "Standardize"


AnyScaler = Optional[tagged_union(Normalize, Standardize)]
