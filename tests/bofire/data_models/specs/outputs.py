from bofire.data_models.domain.api import Outputs
from bofire.data_models.features.api import CategoricalInput, ContinuousOutput
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    Outputs,
    lambda: {
        "features": [
            ContinuousOutput(key="a", objective=None).model_dump(),
            ContinuousOutput(key="b", objective=None).model_dump(),
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

specs.add_invalid(
    Outputs,
    lambda: {
        "features": [
            ContinuousOutput(key="b", objective=None),
            ContinuousOutput(key="b", objective=None),
        ],
    },
    error=ValueError,
    message="Feature keys are not unique.",
)
