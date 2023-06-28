from typing import Union

from bofire.data_models.outlier_detection.outlier_detection import (
    IterativeTrimming,
    OutlierDetection,
)

AnyOutlierDetection = Union[OutlierDetection, IterativeTrimming]
