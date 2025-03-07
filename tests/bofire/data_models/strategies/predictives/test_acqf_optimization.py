import pytest

from bofire.data_models.strategies.api import LSRBO, BotorchOptimizer


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
    acquisition_optimizer = BotorchOptimizer()
    assert acquisition_optimizer.batch_limit == acquisition_optimizer.n_restarts

    acquisition_optimizer = BotorchOptimizer(batch_limit=50)
    assert acquisition_optimizer.batch_limit == acquisition_optimizer.n_restarts
    acquisition_optimizer = BotorchOptimizer(batch_limit=2, n_restarts=4)
    assert acquisition_optimizer.batch_limit == 2
    assert acquisition_optimizer.n_restarts == 4
