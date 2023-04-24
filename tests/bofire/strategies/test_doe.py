import warnings

import numpy as np
import pandas as pd
import pytest

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.strategies.api import DoEStrategy

# from tests.bofire.strategies.botorch.test_model_spec import VALID_MODEL_SPEC_LIST

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

inputs = [
    ContinuousInput(
        key=f"x{1}",
        bounds=(0.0, 1.0),
    ),
    ContinuousInput(
        key=f"x{2}",
        bounds=(0.1, 1.0),
    ),
    ContinuousInput(
        key=f"x{3}",
        bounds=(0.0, 0.6),
    ),
]
domain = Domain.from_lists(
    inputs=inputs,
    outputs=[ContinuousOutput(key="y")],
    constraints=[
        LinearEqualityConstraint(
            features=[f"x{i+1}" for i in range(3)], coefficients=[1, 1, 1], rhs=1
        ),
        LinearInequalityConstraint(features=["x1", "x2"], coefficients=[5, 4], rhs=3.9),
        LinearInequalityConstraint(
            features=["x1", "x2"], coefficients=[-20, 5], rhs=-3
        ),
    ],
)


def test_doe_strategy_init():
    data_model = data_models.DoEStrategy(domain=domain, formula="linear")
    strategy = DoEStrategy(data_model=data_model)
    assert strategy is not None


def test_doe_strategy_ask():
    data_model = data_models.DoEStrategy(domain=domain, formula="linear")
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=12)
    assert candidates.shape == (12, 3)


def test_doe_strategy_ask_with_candidates():
    candidates_fixed = pd.DataFrame(
        np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]]),
        columns=["x1", "x2", "x3"],
    )
    data_model = data_models.DoEStrategy(domain=domain, formula="linear")
    strategy = DoEStrategy(data_model=data_model)
    strategy.set_candidates(candidates_fixed)
    candidates = strategy.ask(candidate_count=12)
    assert candidates.shape == (12, 3)


def test_doe_categoricals_not_implemented():
    categorical_inputs = [
        CategoricalInput(key=f"x{i+1}", categories=["a", "b", "c"]) for i in range(3)
    ]
    domain = Domain.from_lists(
        inputs=categorical_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=[],
    )
    with pytest.raises(Exception):
        data_models.DoEStrategy(domain=domain, formula="linear")


def test_doe_discrete_not_implemented():
    discrete_inputs = [DiscreteInput(key=f"x{i+1}", values=[1, 2, 3]) for i in range(3)]
    domain = Domain.from_lists(
        inputs=discrete_inputs,
        outputs=[ContinuousOutput(key="y")],
        constraints=[],
    )
    with pytest.raises(Exception):
        data_models.DoEStrategy(domain=domain, formula="linear")


def test_nchoosek_implemented():
    nchoosek_constraint = NChooseKConstraint(
        features=[f"x{i+1}" for i in range(3)],
        min_count=0,
        max_count=2,
        none_also_valid=True,
    )
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key=f"x{i+1}", bounds=(0.0, 1.0)) for i in range(3)],
        outputs=[ContinuousOutput(key="y")],
        constraints=[nchoosek_constraint],
    )
    data_model = data_models.DoEStrategy(domain=domain, formula="linear")
    strategy = DoEStrategy(data_model=data_model)
    candidates = strategy.ask(candidate_count=12)
    assert candidates.shape == (12, 3)


def test_formulas_implemented():

    expected_num_candidates = {
        "linear": 7,  # 1+a+b+c+3
        "linear-and-quadratic": 10,  # 1+a+b+c+a**2+b**2+c**2+3
        "linear-and-interactions": 10,  # 1+a+b+c+ab+ac+bc+3
        "fully-quadratic": 13,  # 1+a+b+c+a**2+b**2+c**2+ab+ac+bc+3
    }

    for formula, num_candidates in expected_num_candidates.items():
        data_model = data_models.DoEStrategy(domain=domain, formula=formula)
        strategy = DoEStrategy(data_model=data_model)
        candidates = strategy.ask()
        assert candidates.shape == (num_candidates, 3)


def test_doe_strategy_correctness():
    candidates_fixed = pd.DataFrame(
        np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]]),
        columns=["x1", "x2", "x3"],
    )
    data_model = data_models.DoEStrategy(domain=domain, formula="linear")
    strategy = DoEStrategy(data_model=data_model)
    strategy.set_candidates(candidates_fixed)
    candidates = strategy.ask(candidate_count=12)

    np.random.seed(1)
    candidates_expected = np.array(
        [[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]]
    )
    for row in candidates.to_numpy():
        assert any([np.allclose(row, o, atol=1e-2) for o in candidates_expected])
    for o in candidates_expected[:-1]:
        assert any([np.allclose(o, row, atol=1e-2) for row in candidates.to_numpy()])


def test_doe_strategy_amount_of_candidates():
    candidates_fixed = pd.DataFrame(
        np.array([[0.2, 0.2, 0.6], [0.3, 0.6, 0.1], [0.7, 0.1, 0.2], [0.3, 0.1, 0.6]]),
        columns=["x1", "x2", "x3"],
    )
    data_model = data_models.DoEStrategy(domain=domain, formula="linear")
    strategy = DoEStrategy(data_model=data_model)
    strategy.set_candidates(candidates_fixed)
    candidates = strategy.ask(candidate_count=12)

    np.random.seed(1)
    num_candidates_expected = 12
    assert len(candidates) == num_candidates_expected


# if __name__ == "__main__":
#     test_doe_strategy_ask()
#     test_doe_strategy_ask_with_candidates()
#     test_doe_categoricals_not_implemented()
#     test_doe_discrete_not_implemented()
#     test_nchoosek_implemented()
#     test_formulas_implemented()
#     test_doe_strategy_correctness()
#     test_doe_strategy_amount_of_candidates()
