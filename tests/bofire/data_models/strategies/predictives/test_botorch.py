from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.api import BotorchOptimizer, SoboStrategy


def test_botorch_strategy():
    domain = Domain(
        inputs=[ContinuousInput(key="x", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="y")],
    )
    sobo = SoboStrategy(
        domain=domain,
        acquisition_optimizer=BotorchOptimizer(),
    )
    assert isinstance(sobo.acquisition_optimizer, BotorchOptimizer)
