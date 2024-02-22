from typing import List, Literal, Optional

from bofire.data_models.base import BaseModel


class Transform(BaseModel):
    type: str


class RemoveTransform(Transform):
    type: Literal["RemoveTransition"] = "RemoveTransition"
    to_be_removed_experiments: Optional[List[int]] = None
    to_be_removed_candidates: Optional[List[int]] = None


AnyTransform = RemoveTransform
