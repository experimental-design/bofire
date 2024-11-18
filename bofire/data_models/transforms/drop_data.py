from typing import List, Literal, Optional

from bofire.data_models.transforms.transform import Transform


class DropDataTransform(Transform):
    type: Literal["DropDataTransform"] = "DropDataTransform"
    to_be_removed_experiments: Optional[List[int]] = None
    to_be_removed_candidates: Optional[List[int]] = None
