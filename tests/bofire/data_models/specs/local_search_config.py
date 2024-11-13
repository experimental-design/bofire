from bofire.data_models.strategies.api import LSRBOConfig
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    LSRBOConfig,
    lambda: {
        "gamma": 0.2,
    },
)

specs.add_invalid(LSRBOConfig, lambda: {"gamma": -0.1}, error=ValueError)
