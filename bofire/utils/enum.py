from enum import Enum


class SamplingMethodEnum(Enum):
    UNIFORM = "UNIFORM"
    SOBOL = "SOBOL"
    LHS = "LHS"


class ScalerEnum(Enum):
    """Enumeration class of supported scalers
    Currently, normalization and standardization are implemented.
    """

    NORMALIZE = "NORMALIZE"
    STANDARDIZE = "STANDARDIZE"


class CategoricalMethodEnum(Enum):
    """Enumeration class of supported methods how to handle categorical features
    Currently, exhaustive search and free relaxation are implemented.
    """

    EXHAUSTIVE = "EXHAUSTIVE"
    FREE = "FREE"
    # PR = "PR" available soon


class CategoricalEncodingEnum(Enum):
    """Enumeration class of implemented categorical encodings
    Currently, one-hot and ordinal encoding are implemented.
    """

    ONE_HOT = "ONE_HOT"
    ORDINAL = "ORDINAL"
    DUMMY = "DUMMY"
    DESCRIPTOR = "DESCRIPTOR"  # only possible for categorical with descriptors


class AcquisitionFunctionEnum(Enum):
    QNEI = "QNEI"
    QUCB = "QUCB"
    QEI = "QEI"
    QPI = "QPI"
    QSR = "QSR"


class OutputFilteringEnum(Enum):
    ALL = "ALL"
    ANY = "ANY"


class RegressionMetricsEnum(Enum):
    """Enumeration class for regression metrics."""

    R2 = "R2"
    MAE = "MAE"
    MSD = "MSD"
    MAPE = "MAPE"
