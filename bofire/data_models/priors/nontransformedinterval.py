from typing import Literal

from pydantic import PositiveFloat

from bofire.data_models.priors.constraint import PriorConstraint


class NonTransformedInterval(PriorConstraint):
    """Abstract NonTransformedInterval class.
    Modification of the GPyTorch interval class that does not apply transformations.

    See: https://botorch.readthedocs.io/en/stable/_modules/botorch/utils/constraints.html#NonTransformedInterval

    Attributes:
        lower_bound: The lower bound of the interval.
        upper_bound: The upper bound of the interval.
        initial_value: The initial value within the interval.

    """

    type: Literal["NonTransformedInterval"] = "NonTransformedInterval"
    lower_bound: PositiveFloat
    upper_bound: PositiveFloat
    initial_value: PositiveFloat
