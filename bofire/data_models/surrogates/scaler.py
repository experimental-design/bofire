from enum import Enum
from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.types import NonRestrictedFeatureKeys
from bofire.data_models.unions import tagged_union


class ScalerEnum(str, Enum):
    """Transform of the outputs before fitting: standardization, the logarithm, both,
    or none."""

    STANDARDIZE = "STANDARDIZE"
    IDENTITY = "IDENTITY"
    LOG = "LOG"
    CHAINED_LOG_STANDARDIZE = "CHAINED_LOG_STANDARDIZE"


class Scaler(BaseModel):
    """Scaling of the inputs before they reach the model."""

    type: Any
    features: NonRestrictedFeatureKeys = Field(
        default=[],
        description="Keys of the inputs and engineered features to scale. If empty, "
        "all that reach the model as real values: continuous inputs, categoricals "
        "encoded by static or Mordred descriptors, and all engineered features.",
    )


class Normalize(Scaler):
    """Maps inputs to [0, 1] using their bounds, widened to cover the data; engineered
    features use the range of the data."""

    type: Literal["Normalize"] = "Normalize"


class Standardize(Scaler):
    """Shifts and scales inputs to zero mean and unit variance over the data."""

    type: Literal["Standardize"] = "Standardize"


AnyScaler = Optional[tagged_union(Normalize, Standardize)]
