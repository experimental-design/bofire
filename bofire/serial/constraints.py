from typing import Union

from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)

AnyConstraint = Union[
    # LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    # NonlinearConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    NChooseKConstraint,
]
