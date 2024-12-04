import pandas as pd


class Transform:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def transform_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return experiments

    def transform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates

    def untransform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates
