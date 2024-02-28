from typing import List, Literal, Optional

from pydantic import BaseModel


class RemoveTransform(BaseModel):
    type: Literal["RemoveTransition"] = "RemoveTransition"
    to_be_removed_experiments: Optional[List[int]] = None
    to_be_removed_candidates: Optional[List[int]] = None
