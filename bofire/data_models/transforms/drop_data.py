from typing import List, Literal

from pydantic import Field

from bofire.data_models.transforms.transform import Transform


class DropDataTransform(Transform):
    """Hides some experiments or pending candidates from a step of a stepwise
    strategy."""

    type: Literal["DropDataTransform"] = "DropDataTransform"
    to_be_removed_experiments: List[int] = Field(
        default=[],
        description="Index labels of the experiments the step does not see.",
    )
    to_be_removed_candidates: List[int] = Field(
        default=[],
        description="Index labels of the pending candidates the step does not see.",
    )
