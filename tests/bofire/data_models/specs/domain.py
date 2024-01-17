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
            ]
        ).model_dump(),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(key="o1"),
                features.valid(ContinuousOutput).obj(key="o2"),
            ]
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
            ]
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
            ]
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
            ]
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(key="i1"),
            ]
        ),
    },
    error=ValueError,
    message="Feature keys are not unique",
)
