import pandas as pd

from bofire.data_models.transforms.api import DropDataTransform as DataModel
from bofire.transforms.transform import Transform


class DropDataTransform(Transform):
    def __init__(self, data_model: DataModel):
        self.to_be_removed_experiments = data_model.to_be_removed_experiments or []
        self.to_be_removed_candidates = data_model.to_be_removed_candidates or []

    def transform_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return experiments.drop(self.to_be_removed_experiments)

    def transform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates.drop(self.to_be_removed_candidates)
