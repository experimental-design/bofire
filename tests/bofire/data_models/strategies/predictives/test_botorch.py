import pytest

from bofire.data_models.strategies.api import LSRBO


@pytest.mark.parametrize(
    "gamma, acqf_local, acqf_global, expected",
    [(0.1, 0.3, 0.4, True), (0.4, 0.1, 0.5, False)],
)
def test_LSRBO(gamma, acqf_local, acqf_global, expected):
    assert (
        LSRBO(gamma=gamma).is_local_step(acqf_local=acqf_local, acqf_global=acqf_global)
        == expected
    )
