from enum import Enum
from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel
from bofire.data_models.types import NonRestrictedFeatureKeys
from bofire.data_models.unions import tagged_union


class ScalerEnum(str, Enum):
    """Transform applied to the outputs before fitting; predictions are transformed
    back."""

    STANDARDIZE = "STANDARDIZE"
    """Transform to zero mean and unit variance."""

    IDENTITY = "IDENTITY"
    """Do not transform."""

    LOG = "LOG"
    """Transform to the logarithm; outputs must be positive."""

    CHAINED_LOG_STANDARDIZE = "CHAINED_LOG_STANDARDIZE"
    """Transform to the logarithm, then to zero mean and unit variance."""


class Scaler(BaseModel):
    """Scaling of the inputs before they reach the model."""

    type: Any
    features: NonRestrictedFeatureKeys = Field(
        default=[],
        description="Keys of the inputs and engineered features to scale. If empty, "
        "all of them that reach the model as real values are scaled: continuous "
        "inputs, categoricals encoded by static or Mordred descriptors, and all "
        "engineered features.",
    )


class Normalize(Scaler):
    """Maps the inputs to [0, 1]."""

    type: Literal["Normalize"] = "Normalize"


class Standardize(Scaler):
    """Shifts and scales the inputs to zero mean and unit variance over the data."""

    type: Literal["Standardize"] = "Standardize"


AnyScaler = Optional[tagged_union(Normalize, Standardize)]
