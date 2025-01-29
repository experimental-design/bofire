from bofire.data_models.domain.api import Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalOutput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import ConstrainedCategoricalObjective
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

specs.add_invalid(
    Outputs,
    lambda: {
        "features": [
            CategoricalOutput(
                key="b",
                categories=["a", "b"],
                objective=ConstrainedCategoricalObjective(
                    categories=["c", "d"],
                    desirability=[True, True],
                ),
            ),
        ],
    },
    error=ValueError,
    message="categories must match to objective categories",
)
