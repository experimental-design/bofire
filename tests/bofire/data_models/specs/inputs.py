from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    Inputs,
    lambda: {
        "features": [
            CategoricalInput(
                key="a", categories=["1", "2"], allowed=[True, True]
            ).model_dump(),
            ContinuousInput(key="b", bounds=(0, 1)).model_dump(),
        ],
    },
)


specs.add_invalid(
    Inputs,
    lambda: {
        "features": [
            CategoricalInput(key="a", categories=["1", "2"]),
            ContinuousOutput(key="b", objective=None),
        ],
    },
    error=ValueError,
)
