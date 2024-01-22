from bofire.data_models.strategies.api import LSRBO
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    LSRBO,
    lambda: {
        "gamma": 0.2,
    },
)

specs.add_invalid(LSRBO, lambda: {"gamma": -0.1}, error=ValueError)
