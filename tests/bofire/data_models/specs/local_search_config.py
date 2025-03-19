from bofire.data_models.strategies.predictives.acqf_optimization import LSRBO
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    LSRBO,
    lambda: {
        "gamma": 0.2,
    },
)
