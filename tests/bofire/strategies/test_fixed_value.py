import numpy as np
import pandas as pd
import pytest

import bofire.data_models.strategies.api as data_models
from bofire.data_models.acquisition_functions.api import qLogEI
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.strategies.api import SoboStrategy


def test_fixed_value_respected_in_candidate():
    """Candidate from SoboStrategy must pin the parameter with fixed_value to that value."""
    fixed_val = 3.0

    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0.0, 5.0)),
            ContinuousInput(key="x2", bounds=(0.0, 5.0)),
            ContinuousInput(key="x3", bounds=(0.0, 5.0)),
            ContinuousInput(key="x4", bounds=(1.0, 5.0), fixed_value=fixed_val),
        ],
        outputs=[ContinuousOutput(key="y")],
    )

    rng = np.random.default_rng(42)
    n = 10
    experiments = pd.DataFrame(
        {
            "x1": rng.uniform(0.0, 5.0, n),
            "x2": rng.uniform(0.0, 5.0, n),
            "x3": rng.uniform(0.0, 5.0, n),
            "x4": rng.uniform(1.0, 5.0, n),
            "y": rng.uniform(0.0, 10.0, n),
            "valid_y": [1] * n,
        }
    )

    strategy = SoboStrategy(
        data_model=data_models.SoboStrategy(
            domain=domain,
            acquisition_function=qLogEI(),
        )
    )
    strategy.tell(experiments)

    candidates = strategy.ask(candidate_count=1)

    assert candidates["x4"].iloc[0] == pytest.approx(fixed_val)
