from typing import Union

try:
    from bofire.data_models.outlier_detection.outlier_detection import (
        IterativeTrimming,
        OutlierDetection,
    )

    AnyOutlierDetection = Union[OutlierDetection, IterativeTrimming]
except ImportError:
    pass
