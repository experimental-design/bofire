from typing import List, Literal

from bofire.data_models.constraints.api import LinearConstraint
from bofire.data_models.constraints.constraint import (
    Constraint,
)


class ConstraintGroup(Constraint):

    type: Literal["ConstraintGroup"] = "ConstraintGroup"
    groups: List[List[LinearConstraint]]
