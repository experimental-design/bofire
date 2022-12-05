from enum import Enum


class KernelEnum(Enum):
    """Enumeration class of all supported kernels
    Currently, RBF and matern kernel (1/2, 3/2 and 5/2) are implemented.
    """

    RBF = "RBF"
    MATERN_25 = "MATERN_25"
    MATERN_15 = "MATERN_15"
    MATERN_05 = "MATERN_05"


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


class CategoricalEncodingEnum(Enum):
    """Enumeration class of implemented categorical encodings
    Currently, one-hot and ordinal encoding are implemented.
    """

    ONE_HOT = "ONE_HOT"
    ORDINAL = "ORDINAL"


class DescriptorMethodEnum(Enum):
    """Enumeration class of implemented methods how to handle discrete descriptors
    Currently, exhaustive search and free relaxation are implemented.
    """

    EXHAUSTIVE = "EXHAUSTIVE"
    FREE = "FREE"


class DescriptorEncodingEnum(Enum):
    """Enumeration class how categorical features with descriptors should be encoded
    Categoricals with descriptors can be handled similar to categoricals, or the descriptors can be used.
    """

    DESCRIPTOR = "DESCRIPTOR"
    CATEGORICAL = "CATEGORICAL"


class AcquisitionFunctionEnum(Enum):
    QNEI = "qNEI"
    QUCB = "qUCB"
    QEI = "qEI"
    QPI = "qPI"
    #QSR = "qSR"
    #QEHVI = "qEHVI"
    #QNEHVI = "qNEHVI"
