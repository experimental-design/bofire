import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import bofire.strategies.api as strategies
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.strategies.api import FractionalFactorialStrategy


def test_FractionalFactorialStrategy_ask():
    # standard behavior
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(-2, 8)),
                ],
            ),
        ),
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None).sort_values(by=["a", "b"]).reset_index(drop=True)
    assert len(candidates) == 5
    assert_frame_equal(
        pd.DataFrame(
            {
                "a": [0.0, 1.0, 0.0, 1.0, 0.5],
                "b": [-2.0, -2.0, 8.0, 8.0, 3.0],
            },
        )
        .sort_values(by=["a", "b"])
        .reset_index(drop=True),
        candidates,
    )
    # with repetitions
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(-2, 8)),
                ],
            ),
        ),
        n_repetitions=2,
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None)
    assert len(candidates) == 9
    # with repetitions and 2 center points
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(-2, 8)),
                ],
            ),
        ),
        n_repetitions=2,
        n_center=2,
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None)
    assert len(candidates) == 10
    # no center point, 1 repetition
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(-2, 8)),
                ],
            ),
        ),
        n_repetitions=1,
        n_center=0,
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None)
    assert len(candidates) == 4
    # with number of generators
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    ContinuousInput(key="c", bounds=(0, 1)),
                    ContinuousInput(key="d", bounds=(0, 1)),
                ],
            ),
        ),
        n_repetitions=1,
        n_center=0,
        n_generators=1,
    )
    strategy = strategies.map(strategy_data)
    candidates_auto = strategy.ask(None)
    assert len(candidates_auto) == 8
    # with number of generators
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    ContinuousInput(key="c", bounds=(0, 1)),
                    ContinuousInput(key="d", bounds=(0, 1)),
                ],
            ),
        ),
        n_repetitions=1,
        n_center=0,
        n_generators=1,
        generator="a b c ab",
    )
    strategy = strategies.map(strategy_data)
    candidates_gen = strategy.ask(None)
    with pytest.raises(AssertionError):
        assert_frame_equal(candidates_auto, candidates_gen)
    # here we test a purely categorical space
    strategy_data = FractionalFactorialStrategy(
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
    # here we test a mixed space
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    CategoricalInput(key="alpha", categories=["a", "b"]),
                ],
            ),
        ),
        n_center=1,
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None)
    assert len(candidates) == 10
    # test blocking
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    ContinuousInput(key="c", bounds=(0, 1)),
                    CategoricalInput(key="alpha", categories=["a", "b"]),
                ],
            ),
        ),
        n_center=1,
        n_repetitions=1,
        block_feature_key="alpha",
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None)
    assert len(candidates) == 10
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    ContinuousInput(key="c", bounds=(0, 1)),
                    CategoricalInput(key="alpha", categories=["a", "b"]),
                ],
            ),
        ),
        n_center=0,
        n_repetitions=2,
        block_feature_key="alpha",
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None)
    assert len(candidates) == 16
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(0, 1)),
                    ContinuousInput(key="c", bounds=(0, 1)),
                    CategoricalInput(
                        key="alpha", categories=["a", "b", "c", "d", "e", "f", "g", "h"]
                    ),
                ],
            ),
        ),
        n_center=0,
        n_repetitions=2,
        block_feature_key="alpha",
    )
    strategy = strategies.map(strategy_data)
    candidates = strategy.ask(None)
    assert len(candidates) == 16


def test_FractionalFactorialStrategy_randomize_runorder():
    # test no randomization
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(-2, 8)),
                ],
            ),
        ),
        randomize_runorder=False,
    )
    strategy = strategies.map(strategy_data)
    design = strategy.ask(None)
    design2 = strategy.ask(None)
    # test with randomization
    assert_frame_equal(design, design2)
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(-2, 8)),
                ],
            ),
        ),
        randomize_runorder=True,
        seed=42,
    )
    strategy = strategies.map(strategy_data)
    design = strategy.ask(None)
    design2 = strategy.ask(None)
    with pytest.raises(AssertionError):
        assert_frame_equal(design, design2)
    # test reproducibility with same seed for randomization
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(-2, 8)),
                ],
            ),
        ),
        randomize_runorder=True,
        seed=42,
    )
    strategy = strategies.map(strategy_data)
    design3 = strategy.ask(None)
    design4 = strategy.ask(None)
    with pytest.raises(AssertionError):
        assert_frame_equal(design3, design4)
    assert_frame_equal(design, design3)
    assert_frame_equal(design2, design4)


def test_FractionalFactorialStrategy_ask_invalid():
    strategy_data = FractionalFactorialStrategy(
        domain=Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="a", bounds=(0, 1)),
                    ContinuousInput(key="b", bounds=(-2, 8)),
                ],
            ),
        ),
    )
    strategy = strategies.map(strategy_data)
    with pytest.warns(
        UserWarning,
        match="FractionalFactorialStrategy will ignore the specified value of candidate_count. "
        "The strategy automatically determines how many candidates to "
        "propose.",
    ):
        candidates = strategy.ask(7)
    assert len(candidates) == 5
