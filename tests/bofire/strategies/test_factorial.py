import pytest

import bofire.strategies.api as strategies
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.features.api import CategoricalInput, DiscreteInput
from bofire.data_models.strategies.api import FactorialStrategy


def test_FactorialStrategy_ask():
    strategy_data = FactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    CategoricalInput(key="alpha", categories=["a", "b", "c"]),
                    DiscreteInput(key="beta", values=[1.0, 2, 3.0, 4.0]),
                ],
            ),
        ),
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None)
    assert len(candidates) == 12


def test_FactorialStrategy_ask_invalid():
    strategy_data = FactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    CategoricalInput(key="alpha", categories=["a", "b", "c"]),
                    DiscreteInput(key="beta", values=[1.0, 2, 3.0, 4.0]),
                ],
            ),
        ),
    )
    strategy = strategies.map(strategy_data)
    with pytest.raises(
        ValueError,
        match="FactorialStrategy will ignore the specified value of candidate_count. "
        "The strategy automatically determines how many candidates to "
        "propose.",
    ):
        strategy.ask(5)
