import warnings

import numpy as np
import pandas as pd

import bofire.data_models.strategies.api as data_models
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.strategies.api import DoEStrategy

# from tests.bofire.strategies.botorch.test_model_spec import VALID_MODEL_SPEC_LIST

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

input_features = [
    ContinuousInput(key=f"x{1}", lower_bound=0.0, upper_bound=1.0),
    ContinuousInput(key=f"x{2}", lower_bound=0.1, upper_bound=1.0),
    ContinuousInput(key=f"x{3}", lower_bound=0.0, upper_bound=0.6),
]
domain = Domain(
    input_features=input_features,
    output_features=[ContinuousOutput(key="y")],
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


if __name__ == "__main__":
    test_doe_strategy_amount_of_candidates()
