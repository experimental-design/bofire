import pytest

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.api import LSRBO, SoboStrategy


@pytest.mark.parametrize(
    "gamma, acqf_local, acqf_global, expected",
    [(0.1, 0.3, 0.4, True), (0.4, 0.1, 0.5, False)],
)
def test_LSRBO(gamma, acqf_local, acqf_global, expected):
    assert (
        LSRBO(gamma=gamma).is_local_step(acqf_local=acqf_local, acqf_global=acqf_global)
        == expected
    )


def test_validate_batch_limit():
    domain = Domain(
        inputs=[ContinuousInput(key="a", bounds=(0, 1))],
        outputs=[ContinuousOutput(key="b")],
    )
    strategy_data = SoboStrategy(domain=domain)
    assert strategy_data.batch_limit == strategy_data.num_restarts
    strategy_data = SoboStrategy(domain=domain, batch_limit=50)
    assert strategy_data.batch_limit == strategy_data.num_restarts
    strategy_data = SoboStrategy(domain=domain, batch_limit=2, num_restarts=4)
    assert strategy_data.batch_limit == 2
    assert strategy_data.num_restarts == 4
