import bofire.data_models.means.api as means
from bofire.data_models.priors.api import NormalPrior
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    means.ConstantMean,
    lambda: {"prior": None, "bounds": None},
)
specs.add_valid(
    means.ConstantMean,
    lambda: {
        "prior": NormalPrior(loc=0.0, scale=1.0).model_dump(),
        "bounds": (-10.0, 10.0),
    },
)

specs.add_invalid(
    means.ConstantMean,
    lambda: {"bounds": (1.0, -1.0)},
    error=ValueError,
    message="The lower bound must be less than the upper bound",
)
