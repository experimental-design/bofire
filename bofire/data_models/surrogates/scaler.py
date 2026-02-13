from enum import Enum
from typing import Any, Literal, Optional

from bofire.data_models.base import BaseModel
from bofire.data_models.types import FeatureKeys


class ScalerEnum(str, Enum):
    """Enumeration class of supported scalers
    Currently, log, normalization and standardization are implemented.
    """

    STANDARDIZE = "STANDARDIZE"
    IDENTITY = "IDENTITY"
    LOG = "LOG"
    CHAINED_LOG_STANDARDIZE = "CHAINED_LOG_STANDARDIZE"


class Scaler(BaseModel):
    type: Any
    features: FeatureKeys = []


class Normalize(Scaler):
    type: Literal["Normalize"] = "Normalize"


class Standardize(Scaler):
    type: Literal["Standardize"] = "Standardize"


AnyScaler = Optional[Normalize | Standardize]
