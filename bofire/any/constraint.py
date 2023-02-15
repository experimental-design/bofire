from typing import Union

from bofire.domain import constraints

AnyConstraint = Union[
    constraints.LinearEqualityConstraint,
    constraints.LinearInequalityConstraint,
    constraints.NonlinearEqualityConstraint,
    constraints.NonlinearInequalityConstraint,
    constraints.NChooseKConstraint,
]
