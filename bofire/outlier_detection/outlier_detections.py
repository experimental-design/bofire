from abc import ABC
from typing import List

import pandas as pd

from bofire.data_models.outlier_detection.api import OutlierDetections as DataModel
from bofire.outlier_detection.api import (  # noqa: F401
    IterativeTrimming,
    OutlierDetection,
)
from bofire.outlier_detection.api import map as map_outlier


AnyOutlierDetector = IterativeTrimming


class OutlierDetections(ABC):
    detectors: List[AnyOutlierDetector]

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.detectors = [map_outlier(model) for model in data_model.detectors]  # type: ignore

    def detect(self, experiments: pd.DataFrame) -> pd.DataFrame:
        for outlier_model in self.detectors:
            experiments = outlier_model.detect(experiments)

        return experiments
