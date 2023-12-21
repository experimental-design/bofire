from pydantic.error_wrappers import ValidationError

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
        ),
        "outputs": Outputs(
            features=[
                features.valid(ContinuousOutput).obj(key="o1"),
                features.valid(ContinuousOutput).obj(key="o2"),
            ]
        ),
        "constraints": Constraints(),
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
    error=ValidationError,
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
    error=ValidationError,
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
    error=ValidationError,
)
