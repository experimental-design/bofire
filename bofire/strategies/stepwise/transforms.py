from typing import Optional

import pandas as pd

import bofire.data_models.strategies.stepwise.transforms as data_models
from bofire.data_models.domain.api import Domain


class Transform:
    def transform_experiments(
        self, _experiments: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        pass

    def transform_candidates(self, _candidates: pd.DataFrame) -> Optional[pd.DataFrame]:
        pass

    def transform_domain(self, _domain: Domain) -> Optional[Domain]:
        pass


class RemoveTransform(Transform):
    def __init__(self, data_model: data_models.RemoveTransform):
        self.to_be_removed_candidates = data_model.to_be_removed_candidates
        self.to_be_removed_experiments = data_model.to_be_removed_experiments

    def transform_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return experiments.drop(self.to_be_removed_experiments)

    def transform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates.drop(self.to_be_removed_candidates)


TRANSFORM_MAP = {
    data_models.RemoveTransform: RemoveTransform,
}


def map(
    data_model: data_models.AnyTransform,
) -> RemoveTransform:
    return TRANSFORM_MAP[data_model.__class__](data_model)
