from typing import Union

from bofire.data_models.constraints.categorical import CategoricalExcludeConstraint
from bofire.data_models.constraints.condition import (
    NonZeroCondition,
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
from bofire.data_models.unions import tagged_union


AnyConstraint = tagged_union(
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    NChooseKConstraint,
    InterpointEqualityConstraint,
    ProductEqualityConstraint,
    ProductInequalityConstraint,
    CategoricalExcludeConstraint,
)

AnyContinuousConstraint = tagged_union(
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
    NChooseKConstraint,
    InterpointEqualityConstraint,
    ProductEqualityConstraint,
    ProductInequalityConstraint,
)

AnyCategoricalConstraint = CategoricalExcludeConstraint

AnyCondition = tagged_union(
    SelectionCondition,
    ThresholdCondition,
    NonZeroCondition,
)

AnyConstraintError = Union[ConstraintError, ConstraintNotFulfilledError]
