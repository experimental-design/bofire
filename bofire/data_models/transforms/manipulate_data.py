from typing import List, Literal, Optional

from pydantic import Field, model_validator

from bofire.data_models.transforms.transform import Transform


class ManipulateDataTransform(Transform):
    """Transform that can be used to manipulate experiments/candidates by applying pandas based transformations
    as described here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.eval.html#pandas.DataFrame.eval

    Attributes:
        experiment_transformations: List of strings representing the transformations to be applied to the experiments
        candidate_transformations: List of strings representing the transformations to be applied to the candidates
        candidate_untransformations: List of strings representing the transformations to be applied to untransform the
            generated candidates

    """

    type: Literal["ManipulateDataTransform"] = "ManipulateDataTransform"
    experiment_transforms: Optional[List[str]] = Field(None, min_length=1)
    candidate_transforms: Optional[List[str]] = Field(None, min_length=1)
    candidate_untransforms: Optional[List[str]] = Field(None, min_length=1)

    @model_validator(mode="after")
    def validate_transformations(self):
        if not any(
            [
                self.experiment_transforms,
                self.candidate_transforms,
                self.candidate_untransforms,
            ]
        ):
            raise ValueError(
                "At least one of experiment_transforms, candidate_transforms, or candidate_untransforms must be provided."
            )

        return self
