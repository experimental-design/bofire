from bofire.benchmarks.detergent import Detergent
from bofire.data_models.strategies.random import (
    RandomStrategy as RandomStrategyDataModel,
)
from bofire.strategies.api import RandomStrategy


def test_detergent():
    d = Detergent()
    assert len(d.domain.inputs) == 5
    assert len(d.domain.outputs) == 5
    assert len(d.domain.constraints) == 2

    # check that the domain has feasible points
    random_search_dm = RandomStrategyDataModel(domain=d.domain)
    random_search = RandomStrategy(random_search_dm)
    candidates = random_search.ask(2)
    assert d.domain.constraints.is_fulfilled(candidates).all()
    y = d.f(candidates)
    assert y.shape == (2, 5)
    for o_key in d.domain.outputs.get_keys():
        assert o_key in y.columns
