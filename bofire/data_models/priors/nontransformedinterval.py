from typing import Literal

from pydantic import PositiveFloat

from bofire.data_models.priors.constraint import PriorConstraint


class NonTransformedInterval(PriorConstraint):
    """Abstract NonTransformedInterval class.
    Modification of the GPyTorch interval class that does not apply transformations.

    See: https://botorch.readthedocs.io/en/stable/_modules/botorch/utils/constraints.html#NonTransformedInterval

    Attributes:
        lower_bound(PositiveFloat): The lower bound of the interval.
        upper_bound(PositiveFloat): The upper bound of the interval.
        initial_value(PositiveFloat): The initial value within the interval.

    """

    type: Literal["NonTransformedInterval"] = "NonTransformedInterval"
    lower_bound: PositiveFloat
    upper_bound: PositiveFloat
    initial_value: PositiveFloat
