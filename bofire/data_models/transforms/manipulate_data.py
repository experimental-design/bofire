from typing import List, Literal, Optional

from pydantic import Field, model_validator

from bofire.data_models.transforms.transform import Transform


class ManipulateDataTransform(Transform):
    """Rewrites columns of the experiments and candidates of a step of a stepwise
    strategy with pandas expressions, as `DataFrame.eval` evaluates them, e.g.
    `"y = y * 100"`.

    At least one of the three lists has to be given.
    """

    type: Literal["ManipulateDataTransform"] = "ManipulateDataTransform"
    experiment_transforms: Optional[List[str]] = Field(
        None,
        min_length=1,
        description="Expressions applied to the experiments before the step sees "
        "them.",
    )
    candidate_transforms: Optional[List[str]] = Field(
        None,
        min_length=1,
        description="Expressions applied to the pending candidates before the step "
        "sees them.",
    )
    candidate_untransforms: Optional[List[str]] = Field(
        None,
        min_length=1,
        description="Expressions applied to the candidates the step proposes, to "
        "undo the transformation.",
    )

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
