from bofire.data_models.strategies.api import (
    AbsoluteMovingReferenceValue,
    FixedReferenceValue,
    MoboStrategy,
)
from tests.bofire.data_models.specs.api import domain


def test_ref_point_generation():
    d = domain.valid().obj()
    strategy_data = MoboStrategy(domain=d)
    for value in strategy_data.ref_point.values.values():
        assert value == AbsoluteMovingReferenceValue(orient_at_best=False, offset=0.0)

    strategy_data = MoboStrategy(domain=d, ref_point={"o1": 6, "o2": 10})
    assert strategy_data.ref_point.values["o1"] == FixedReferenceValue(value=6)
    assert strategy_data.ref_point.values["o2"] == FixedReferenceValue(value=10)
