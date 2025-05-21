from typing import Union

from bofire.data_models.constraints.categorical import (
    CategoricalExcludeConstraint,
    SelectionCondition,
    ThresholdCondition,
)
from bofire.data_models.constraints.constraint import (
    Constraint,
    ConstraintError,
    ConstraintNotFulfilledError,
    EqualityConstraint,
    InequalityConstraint,
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
from bofire.data_models.constraints.product import (
    ProductConstraint,
    ProductEqualityConstraint,
    ProductInequalityConstraint,
)


AbstractConstraint = Union[
    Constraint,
    LinearConstraint,
    NonlinearConstraint,
    IntrapointConstraint,
    InterpointConstraint,
    ProductConstraint,
    InequalityConstraint,
    EqualityConstraint,
]

AnyConstraint = Union[
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    NChooseKConstraint,
    InterpointEqualityConstraint,
    ProductEqualityConstraint,
    ProductInequalityConstraint,
    CategoricalExcludeConstraint,
]

AnyContinuousConstraint = Union[
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    NChooseKConstraint,
    InterpointEqualityConstraint,
    ProductEqualityConstraint,
    ProductInequalityConstraint,
]

AnyCategoricalConstraint = CategoricalExcludeConstraint

AnyCondition = Union[
    SelectionCondition,
    ThresholdCondition,
]

AnyConstraintError = Union[ConstraintError, ConstraintNotFulfilledError]
