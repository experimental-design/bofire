from typing import Union

from bofire.domain import constraint

AnyConstraint = Union[
    constraint.LinearEqualityConstraint,
    constraint.LinearInequalityConstraint,
    constraint.NonlinearEqualityConstraint,
    constraint.NonlinearInequalityConstraint,
    constraint.NChooseKConstraint,
]
