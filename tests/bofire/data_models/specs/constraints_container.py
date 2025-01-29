from bofire.data_models.constraints.api import LinearInequalityConstraint
from bofire.data_models.domain.api import Constraints
from bofire.data_models.features.api import CategoricalInput
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    Constraints,
    lambda: {
        "constraints": [
            LinearInequalityConstraint(
                features=["a", "b"],
                coefficients=[1, 1],
                rhs=1,
            ).model_dump(),
        ],
    },
)


specs.add_invalid(
    Constraints,
    lambda: {
        "constraints": [
            CategoricalInput(key="a", categories=["1", "2"], allowed=[True, True]),
        ],
    },
    error=ValueError,
)

specs.add_invalid(
    Constraints,
    lambda: {
        "constraints": [
            LinearInequalityConstraint(features=["a", "b"], coefficients=[1, 1], rhs=1),
            CategoricalInput(key="a", categories=["1", "2"], allowed=[True, True]),
        ],
    },
    error=ValueError,
)

specs.add_invalid(
    Constraints,
    lambda: {
        "constraints": [
            LinearInequalityConstraint(features=["a", "b"], coefficients=[1, 1], rhs=1),
            "s",
        ],
    },
    error=ValueError,
)
