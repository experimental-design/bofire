from typing import Literal

from pydantic import Field, PositiveFloat

from bofire.data_models.priors.gamma import GammaPrior
from bofire.data_models.priors.prior import Prior


class LKJPrior(Prior):
    """LKJ prior over correlation matrices. Allows to specify the shape of the prior.

    Applies to the correlation matrix between the tasks of a multi-task GP, rather
    than to a single scalar hyperparameter.

    Examples:
        >>> LKJPrior(shape=2.0, sd_prior=GammaPrior(concentration=2.0, rate=0.15))
    """

    type: Literal["LKJPrior"] = "LKJPrior"
    shape: PositiveFloat = Field(
        description="Shape parameter of the LKJ distribution. Larger values "
        "concentrate the prior on correlation matrices closer to the identity.",
    )
    sd_prior: GammaPrior = Field(
        description="Prior over the standard deviations of the correlation matrix.",
    )
    n_tasks: int = Field(
        default=2,
        description="Number of dimensions of the correlation matrix, i.e. the number "
        "of tasks.",
    )
