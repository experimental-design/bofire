from enum import Enum


class ScalerEnum(str, Enum):
    """Enumeration class of supported scalers
    Currently, log, normalization and standardization are implemented.
    """

    NORMALIZE = "NORMALIZE"
    STANDARDIZE = "STANDARDIZE"
    IDENTITY = "IDENTITY"
    LOG = "LOG"
    CHAINED_LOG_STANDARDIZE = "CHAINED_LOG_STANDARDIZE"
