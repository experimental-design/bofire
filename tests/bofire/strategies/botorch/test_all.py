import pytest
from pydantic.error_wrappers import ValidationError

from bofire.strategies.botorch.qehvi import BoTorchQehviStrategy, BoTorchQnehviStrategy
from bofire.strategies.botorch.qparego import BoTorchQparegoStrategy
from bofire.strategies.botorch.sobo import (
    BoTorchSoboAdditiveStrategy,
    BoTorchSoboMultiplicativeStrategy,
    BoTorchSoboStrategy,
)
from tests.bofire.domain.test_domain_validators import generate_experiments
from tests.bofire.strategies.botorch.test_base import domains
from tests.bofire.strategies.botorch.test_qehvi import BOTORCH_QEHVI_STRATEGY_SPECS
from tests.bofire.strategies.botorch.test_qparego import BOTORCH_QPAREGO_STRATEGY_SPECS
from tests.bofire.strategies.botorch.test_sobo import BOTORCH_SOBO_STRATEGY_SPECS

STRATEGY_SPECS = {
    BoTorchSoboStrategy: BOTORCH_SOBO_STRATEGY_SPECS,
    BoTorchSoboAdditiveStrategy: BOTORCH_SOBO_STRATEGY_SPECS,
    BoTorchSoboMultiplicativeStrategy: BOTORCH_SOBO_STRATEGY_SPECS,
    BoTorchQehviStrategy: BOTORCH_QEHVI_STRATEGY_SPECS,
    BoTorchQnehviStrategy: BOTORCH_QEHVI_STRATEGY_SPECS,
    BoTorchQparegoStrategy: BOTORCH_QPAREGO_STRATEGY_SPECS,
}
#TODO: add ModelPredictiveStrategy

@pytest.mark.parametrize(
    "cls, spec",
    [(cls, valid) for cls, data in STRATEGY_SPECS.items() for valid in data["valids"]],
)
def test_valid_strategy_specs(cls, spec):
    res = cls(**spec)
    assert isinstance(res, cls)


@pytest.mark.parametrize(
    "cls, spec",
    [
        (cls, invalid)
        for cls, data in STRATEGY_SPECS.items()
        for invalid in data["invalids"]
    ],
)
def test_invalid_strategy_specs(cls, spec):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        res = cls(**spec)


# TODO: add per-strategy test for strategy.has_sufficient_experiments
# TODO: add per-strategy test for strategy.tell
