import bofire.data_models.means.api as means
from bofire.data_models.priors.api import GreaterThan, NormalPrior
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    means.ConstantMean,
    lambda: {"prior": None, "constraint": None},
)
specs.add_valid(
    means.ConstantMean,
    lambda: {
        "prior": NormalPrior(loc=0.0, scale=1.0).model_dump(),
        "constraint": GreaterThan(lower_bound=-10.0).model_dump(),
    },
)
