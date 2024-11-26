from enum import Enum


class SamplingMethodEnum(str, Enum):
    UNIFORM = "UNIFORM"
    SOBOL = "SOBOL"
    LHS = "LHS"


class CategoricalMethodEnum(str, Enum):
    """Enumeration class of supported methods how to handle categorical features
    Currently, exhaustive search and free relaxation are implemented.
    """

    EXHAUSTIVE = "EXHAUSTIVE"
    FREE = "FREE"
    # PR = "PR" available soon


class CategoricalEncodingEnum(str, Enum):
    """Enumeration class of implemented categorical encodings
    Currently, one-hot and ordinal encoding are implemented.
    """

    ONE_HOT = "ONE_HOT"
    ORDINAL = "ORDINAL"
    DUMMY = "DUMMY"
    DESCRIPTOR = "DESCRIPTOR"  # only possible for categorical with descriptors


class ClassificationMetricsEnum(str, Enum):
    """Enumeration class for classification metrics."""

    ACCURACY = "ACCURACY"
    F1 = "F1"


class OutputFilteringEnum(str, Enum):
    ALL = "ALL"
    ANY = "ANY"


class RegressionMetricsEnum(str, Enum):
    """Enumeration class for regression metrics."""

    R2 = "R2"
    MAE = "MAE"
    MSD = "MSD"
    MAPE = "MAPE"
    PEARSON = "PEARSON"
    SPEARMAN = "SPEARMAN"
    FISHER = "FISHER"


class UQRegressionMetricsEnum(str, Enum):
    """Enumeration class for ucertainty regression metrics."""

    PEARSON_UQ = "PEARSON_UQ"
    SPEARMAN_UQ = "SPEARMAN_UQ"
    KENDALL_UQ = "KENDALL_UQ"
    MAXIMUMCALIBRATION = "MAXIMUMCALIBRATION"
    MISCALIBRATIONAREA = "MISCALIBRATIONAREA"
    ABSOLUTEMISCALIBRATIONAREA = "ABSOLUTEMISCALIBRATIONAREA"
