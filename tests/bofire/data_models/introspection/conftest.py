import pytest

from bofire.data_models.constraints.api import LinearInequalityConstraint
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import RBFKernel
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.strategies.api import SoboStrategy
from bofire.data_models.surrogates.api import BotorchSurrogates, SingleTaskGPSurrogate


@pytest.fixture
def domain() -> Domain:
    return Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="temperature", bounds=(20, 150), unit="degC"),
                ContinuousInput(key="pressure", bounds=(1, 10), unit="bar"),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="yield", objective=MaximizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                LinearInequalityConstraint(
                    features=["temperature", "pressure"], coefficients=[1, 1], rhs=200
                )
            ]
        ),
    )


@pytest.fixture
def spec(domain) -> dict:
    """A strategy spec whose surrogate_specs were auto-derived."""
    return SoboStrategy(domain=domain).model_dump(mode="json")


@pytest.fixture
def custom_spec(domain) -> dict:
    """A strategy spec the user customised, so nothing may be regenerated."""
    return SoboStrategy(
        domain=domain,
        surrogate_specs=BotorchSurrogates(
            surrogates=[
                SingleTaskGPSurrogate(
                    inputs=domain.inputs,
                    outputs=domain.outputs,
                    kernel=RBFKernel(ard=False),
                )
            ]
        ),
    ).model_dump(mode="json")
