from typing import Union

from bofire.data_models.outlier_detection.outlier_detection import (
    IterativeTrimming,
    OutlierDetection,
)
from bofire.data_models.outlier_detection.outlier_detections import OutlierDetections


AnyOutlierDetection = IterativeTrimming
AnyAbstractOutlierDetection = OutlierDetection
