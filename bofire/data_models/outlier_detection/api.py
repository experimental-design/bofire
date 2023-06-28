from typing import Union
try:
    from bofire.data_models.outlier_detection.outlier_detection import OutlierDetection,IterativeTrimming
    AnyOutlierDetection = Union[OutlierDetection,IterativeTrimming]
except ImportError:
    pass
  