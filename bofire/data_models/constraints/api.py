from typing import Union

from bofire.data_models.constraints.categorical import CategoricalExcludeConstraint
from bofire.data_models.constraints.condition import (
    NonZeroCondition,
    SelectionCondition,
    ThresholdCondition,
)
from bofire.data_models.constraints.constraint import (
    Constraint,  # noqa: F401 re-export
    ConstraintError,
    ConstraintNotFulfilledError,
)
from bofire.data_models.constraints.interpoint import (
    InterpointConstraint,  # noqa: F401 re-export
    InterpointEqualityConstraint,
)
from bofire.data_models.constraints.linear import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.constraints.nchoosek import NChooseKConstraint
from bofire.data_models.constraints.nonlinear import (
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.constraints.product import (
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
