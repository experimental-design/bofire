from enum import Enum


class SamplingMethodEnum(Enum):
    UNIFORM = "UNIFORM"
    SOBOL = "SOBOL"
    LHS = "LHS"


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


class OutputFilteringEnum(Enum):
    ALL = "ALL"
    ANY = "ANY"


class RegressionMetricsEnum(Enum):
    """Enumeration class for regression metrics."""

    R2 = "R2"
    MAE = "MAE"
    MSD = "MSD"
    MAPE = "MAPE"
    PEARSON = "PEARSON"
    SPEARMAN = "SPEARMAN"
    FISHER = "FISHER"


class UQRegressionMetricsEnum(Enum):
    """Enumeration class for ucertainty regression metrics."""

    PEARSON_UQ = "PEARSON_UQ"
    SPEARMAN_UQ = "SPEARMAN_UQ"
    KENDALL_UQ = "KENDALL_UQ"
    MAXIMUMCALIBRATION = "MAXIMUMCALIBRATION"
    MISCALIBRATIONAREA = "MISCALIBRATIONAREA"
    ABSOLUTEMISCALIBRATIONAREA = "ABSOLUTEMISCALIBRATIONAREA"
