from typing import List, Literal, Optional

import pandas as pd

from bofire.data_models.base import BaseModel
from bofire.data_models.domain.api import Domain


class Transform(BaseModel):
    type: str

    def transform_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return experiments

    def transform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates

    def transform_domain(self, domain: Domain) -> Domain:
        return domain


class RemoveTransform(Transform):
    type: Literal["RemoveTransition"] = "RemoveTransition"
    to_be_removed_experiments: Optional[List[int]] = None
    to_be_removed_candidates: Optional[List[int]] = None

    def transform_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return experiments.drop(self.to_be_removed_experiments)

    def transform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates.drop(self.to_be_removed_candidates)


AnyTransform = RemoveTransform
