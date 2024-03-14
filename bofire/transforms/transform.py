import pandas as pd

from bofire.data_models.domain.api import Domain


class Transform:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def transform_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return experiments

    def transform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates

    def transform_domain(self, domain: Domain) -> Domain:
        return domain

    def untransform_candidates(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return experiments
