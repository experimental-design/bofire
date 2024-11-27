import pandas as pd

from bofire.data_models.transforms.api import ManipulateDataTransform as DataModel
from bofire.transforms.transform import Transform


class ManipulateDataTransform(Transform):
    def __init__(self, data_model: DataModel):
        self.experiment_transforms = data_model.experiment_transforms or []
        self.candidate_transforms = data_model.candidate_transforms or []
        self.candidate_untransforms = data_model.candidate_untransforms or []

    def _apply_pd_transforms(self, df: pd.DataFrame, transforms: list) -> pd.DataFrame:
        if len(transforms) == 0:
            return df
        transformed_df = df.copy()
        for tr in transforms:
            transformed_df.eval(tr, inplace=True)

        return transformed_df

    def transform_experiments(self, experiments: pd.DataFrame) -> pd.DataFrame:
        return self._apply_pd_transforms(experiments, self.experiment_transforms)

    def transform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return self._apply_pd_transforms(candidates, self.candidate_transforms)

    def untransform_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return self._apply_pd_transforms(candidates, self.candidate_untransforms)
