from typing import Union

from bofire.data_models.constraints.constraint import (
    Constraint,
    ConstraintError,
    ConstraintNotFulfilledError,
    IntrapointConstraint,
)
from bofire.data_models.constraints.interpoint import (
    InterpointConstraint,
    InterpointEqualityConstraint,
)
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
    IntrapointConstraint,
    InterpointConstraint,
]

AnyConstraint = Union[
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    NChooseKConstraint,
    InterpointEqualityConstraint,
]

AnyConstraintError = Union[ConstraintError, ConstraintNotFulfilledError]

# %%
