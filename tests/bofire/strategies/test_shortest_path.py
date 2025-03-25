import numpy as np
import pytest

import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from tests.bofire.data_models.specs.strategies import specs  # as priors


def test_map():
    data_model = specs.valid(data_models.ShortestPathStrategy).obj()
    strategy = strategies.map(data_model=data_model)
    assert isinstance(strategy, strategies.ShortestPathStrategy)


def test_get_linear_constraints():
    data_model = specs.valid(data_models.ShortestPathStrategy).obj()
    strategy = strategies.map(data_model=data_model)
    A, b = strategy.get_linear_constraints(
        data_model.domain.constraints.get(LinearEqualityConstraint),
    )
    assert np.allclose(b, np.array([0.9]))
    assert np.allclose(A, np.array([[1.0, 1.0, 0.0]]))
    A, b = strategy.get_linear_constraints(
        data_model.domain.constraints.get(LinearInequalityConstraint),
    )
    assert np.allclose(b, np.array([0.95]))
    assert np.allclose(A, np.array([[1.0, 1.0, 0.0]]))


def test_step():
    data_model = specs.valid(data_models.ShortestPathStrategy).obj()
    strategy = strategies.map(data_model=data_model)
    step = strategy.step(strategy.start)
    assert step.d == "b"
    assert np.allclose(step[["a", "b", "c"]].tolist(), [0.7, 0.2, 0.1])


def test_ask():
    data_model = specs.valid(data_models.ShortestPathStrategy).obj()
    strategy = strategies.map(data_model=data_model)
    with pytest.warns(
        UserWarning, match="ShortestPathStrategy will ignore the specified "
    ):
        strategy.ask(candidate_count=4)
    steps = strategy.ask()
    assert np.allclose(
        steps.iloc[-1][["a", "b", "c"]].tolist(),
        strategy.end[["a", "b", "c"]].tolist(),
    )
