from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from tests.bofire.data_models.specs.features import specs as features
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])
specs.add_valid(
    Domain,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(key="i1"),
                features.valid(ContinuousInput).obj(key="i2"),
                features.valid(ContinuousInput).obj(key="i3"),
            ],
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(key="o1"),
                features.valid(ContinuousOutput).obj(key="o2"),
            ],
        ).model_dump(),
        "constraints": Constraints().model_dump(),
    },
)


# duplicate feature names
specs.add_invalid(
    Domain,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(key="i1"),
                features.valid(ContinuousInput).obj(key="i1"),
            ],
        ),
    },
    error=ValueError,
    message="Feature keys are not unique",
)

specs.add_invalid(
    Domain,
    lambda: {
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(key="i1"),
                features.valid(ContinuousOutput).obj(key="i1"),
            ],
        ),
    },
    error=ValueError,
    message="Feature keys are not unique",
)

specs.add_invalid(
    Domain,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(key="i1"),
            ],
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(key="i1"),
            ],
        ),
    },
    error=ValueError,
    message="Feature keys are not unique",
)

specs.add_invalid(
    Domain,
    lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(key="i1"),
                features.valid(ContinuousInput).obj(key="i2"),
            ],
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(key="o1"),
            ],
        ),
        "constraints": Constraints(
            constraints=[InterpointEqualityConstraint(features=["i3"])],
        ),
    },
    error=ValueError,
    message="Feature i3 is not a continuous input feature in the provided Inputs object.",
)


def create_spec(c):
    return lambda: {
        "inputs": Inputs(
            features=[
                features.valid(ContinuousInput).obj(key="i1"),
                features.valid(ContinuousInput).obj(key="i2"),
            ],
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(key="o1"),
            ],
        ),
        "constraints": Constraints(constraints=[c]),
    }


for c in [
    LinearInequalityConstraint(
        features=["i1", "i2", "i3"],
        coefficients=[1, 2, 3],
        rhs=1.0,
    ),
    NChooseKConstraint(
        features=["i1", "i2", "i3"],
        min_count=1,
        max_count=1,
        none_also_valid=False,
    ),
    NonlinearInequalityConstraint(features=["i1", "i2", "i3"], expression="i1*i2"),
    ProductInequalityConstraint(
        features=["i1", "i2", "i3"],
        exponents=[1, 1, 1],
        rhs=0,
        sign=1,
    ),
]:
    specs.add_invalid(
        Domain,
        create_spec(c),
        error=ValueError,
        message="Feature i3 is not a continuous input feature in the provided Inputs object.",
    )
