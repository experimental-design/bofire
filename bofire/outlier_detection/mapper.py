from typing import Dict, Type

from bofire.data_models.outlier_detection import api as data_models
from bofire.outlier_detection.outlier_detection import (
    IterativeTrimming,
    OutlierDetection,
)


OUTLIER_MAP: Dict[Type[data_models.OutlierDetection], Type[OutlierDetection]] = {
    data_models.IterativeTrimming: IterativeTrimming,
}


def map(data_model: data_models.OutlierDetection) -> OutlierDetection:
    cls = OUTLIER_MAP[data_model.__class__]
    return cls(data_model=data_model)  # type: ignore
