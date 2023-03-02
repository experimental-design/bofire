from typing import Union

from bofire.any.constraint import AnyConstraint
from bofire.any.constraints import AnyConstraints
from bofire.any.domain import AnyDomain
from bofire.any.feature import AnyFeature
from bofire.any.features import AnyFeatures
from bofire.any.kernel import AnyKernel
from bofire.any.model import AnyModel
from bofire.any.objective import AnyObjective
from bofire.any.prior import AnyPrior
from bofire.any.sampler import AnySampler
from bofire.any.strategy import AnyStrategy

AnyThing = Union[
    AnyConstraint,
    AnyDomain,
    AnyFeature,
    AnyModel,
    AnyObjective,
    AnyStrategy,
    AnyKernel,
    AnyPrior,
    AnyConstraints,
    AnyFeatures,
    AnySampler,
]
