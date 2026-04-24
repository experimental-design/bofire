from typing import Annotated, Union

from pydantic import Field

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

AnyConstraint = Annotated[
    Union[
        LinearEqualityConstraint,
        LinearInequalityConstraint,
        NonlinearEqualityConstraint,
        NonlinearInequalityConstraint,
        NChooseKConstraint,
        InterpointEqualityConstraint,
        ProductEqualityConstraint,
        ProductInequalityConstraint,
        CategoricalExcludeConstraint,
    ],
    Field(discriminator="type"),
]

AnyContinuousConstraint = Annotated[
    Union[
        LinearEqualityConstraint,
        LinearInequalityConstraint,
        NonlinearEqualityConstraint,
        NonlinearInequalityConstraint,
        NChooseKConstraint,
        InterpointEqualityConstraint,
        ProductEqualityConstraint,
        ProductInequalityConstraint,
    ],
    Field(discriminator="type"),
]

AnyCategoricalConstraint = CategoricalExcludeConstraint

AnyCondition = Annotated[
    Union[
        SelectionCondition,
        ThresholdCondition,
        NonZeroCondition,
    ],
    Field(discriminator="type"),
]

AnyConstraintError = Union[ConstraintError, ConstraintNotFulfilledError]
