import pandas as pd

from bofire.data_models.constraints.api import (
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.strategies.api import SoboStrategy


def test_sobo_with_nonlinear_inequality():
    """Test that SoboStrategy can handle nonlinear inequality constraints."""

    # Create a simple domain with 2 inputs and a nonlinear constraint
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        constraints=[
            NonlinearInequalityConstraint(
                expression="x1**2 + x2**2 - 0.5", features=["x1", "x2"]
            )
        ],
    )

    # ✅ CORRECT INITIALIZATION
    strategy = SoboStrategy.make(domain=domain)

    # Add some initial data
    experiments = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3],
            "x2": [0.1, 0.2, 0.3],
            "y": [0.5, 0.6, 0.7],
            "valid_y": [1, 1, 1],
        }
    )

    strategy.tell(experiments)

    # Try to ask for new candidates - THIS is where the real error should appear
    candidates = strategy.ask(1)

    print(candidates)
    assert len(candidates) == 1


def test_sobo_with_nonlinear_equality():
    """Test that SoboStrategy can handle nonlinear equality constraints."""

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=(0, 1)),
                ContinuousInput(key="x2", bounds=(0, 1)),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        constraints=[
            NonlinearEqualityConstraint(
                expression="x1 + x2 - 0.7", features=["x1", "x2"]
            )
        ],
    )

    # ⚠️ IMPORTANT: Set batch_limit=1 for nonlinear constraints
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    experiments = pd.DataFrame(
        {
            "x1": [0.3, 0.4, 0.35],
            "x2": [0.4, 0.3, 0.35],
            "y": [0.5, 0.6, 0.55],
            "valid_y": [1, 1, 1],
        }
    )

    strategy.tell(experiments)

    # Ask for 1 candidate at a time (required for nonlinear constraints)
    candidates = strategy.ask(1)

    # Verify constraint satisfaction
    x1, x2 = candidates.iloc[0]["x1"], candidates.iloc[0]["x2"]
    assert (
        abs((x1 + x2) - 0.7) < 0.01
    ), f"Equality constraint violated: {x1} + {x2} = {x1+x2}"

    print(f"✅ Candidate: x1={x1:.4f}, x2={x2:.4f}, sum={x1+x2:.4f}")
