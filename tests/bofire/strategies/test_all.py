import pytest
from pydantic.error_wrappers import ValidationError

import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from tests.bofire.strategies.test_qehvi import BOTORCH_QEHVI_STRATEGY_SPECS
from tests.bofire.strategies.test_qparego import BOTORCH_QPAREGO_STRATEGY_SPECS
from tests.bofire.strategies.test_sobo import (
    BOTORCH_ADDITIVE_AND_MULTIPLICATIVE_SOBO_STRATEGY_SPECS,
    BOTORCH_SOBO_STRATEGY_SPECS,
)

STRATEGY_SPECS = {
    data_models.SoboStrategy: BOTORCH_SOBO_STRATEGY_SPECS,
    data_models.AdditiveSoboStrategy: BOTORCH_ADDITIVE_AND_MULTIPLICATIVE_SOBO_STRATEGY_SPECS,
    data_models.MultiplicativeSoboStrategy: BOTORCH_ADDITIVE_AND_MULTIPLICATIVE_SOBO_STRATEGY_SPECS,
    data_models.QehviStrategy: BOTORCH_QEHVI_STRATEGY_SPECS,
    data_models.QnehviStrategy: BOTORCH_QEHVI_STRATEGY_SPECS,
    data_models.QparegoStrategy: BOTORCH_QPAREGO_STRATEGY_SPECS,
}


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
        print("mycls:", cls)
        print("invalid spec:", spec)
        data_model = cls(**spec)
        strategies.map(data_model=data_model)


# TODO: add per-strategy test for strategy.has_sufficient_experiments
# TODO: add per-strategy test for strategy.tell
