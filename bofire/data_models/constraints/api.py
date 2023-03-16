from typing import Union

from bofire.data_models.constraints.constraint import Constraint
from bofire.data_models.constraints.linear import (
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.constraints.nchoosek import NChooseKConstraint
from bofire.data_models.constraints.nonlinear import (
    NonlinearConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)

AbstractConstraint = Union[
    Constraint,
    LinearConstraint,
    NonlinearConstraint,
]

AnyConstraint = Union[
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    NChooseKConstraint,
]
