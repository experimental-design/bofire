from bofire.data_models.domain.api import Outputs
from bofire.data_models.features.api import CategoricalInput, ContinuousOutput
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    Outputs,
    lambda: {
        "features": [
            ContinuousOutput(key="a", objective=None),
            ContinuousOutput(key="b", objective=None),
        ],
    },
)


specs.add_invalid(
    Outputs,
    lambda: {
        "features": [
            CategoricalInput(key="a", categories=["1", "2"]),
            ContinuousOutput(key="b", objective=None),
        ],
    },
    error=ValueError,
)
