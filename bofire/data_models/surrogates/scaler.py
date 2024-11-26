from enum import Enum


class ScalerEnum(str, Enum):
    """Enumeration class of supported scalers
    Currently, normalization and standardization are implemented.
    """

    NORMALIZE = "NORMALIZE"
    STANDARDIZE = "STANDARDIZE"
    IDENTITY = "IDENTITY"
