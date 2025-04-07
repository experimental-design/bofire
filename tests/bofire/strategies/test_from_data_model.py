from bofire.strategies import api as strategies


def test_strategy_can_be_loaded_from_data_model(strategy_spec):
    data_model = strategy_spec.obj()
    strategy = strategies.map(data_model=data_model)
    assert strategy is not None
