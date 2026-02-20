import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import NonlinearEqualityConstraint
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective


def test_nonlinear_equality_constraint_sobo():
    """Test that SoboStrategy handles NonlinearEqualityConstraint correctly."""

    # Import the correct modules
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    # Create domain with nonlinear equality constraint
    inputs = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
        ]
    )

    outputs = Outputs(
        features=[ContinuousOutput(key="y", objective=MaximizeObjective())]
    )

    constraints = [
        NonlinearEqualityConstraint(expression="x1 + x2 - 0.7", features=["x1", "x2"])
    ]

    domain = Domain(inputs=inputs, outputs=outputs, constraints=constraints)

    # Create data model first, then strategy
    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    # Add dummy experiments
    experiments = pd.DataFrame(
        {
            "x1": [0.3, 0.4, 0.5],
            "x2": [0.4, 0.3, 0.2],
            "y": [0.5, 0.6, 0.7],
            "valid_y": [1, 1, 1],
        }
    )

    strategy.tell(experiments)

    # Ask for candidates
    candidates = strategy.ask(1)
    # In your test file, after creating the domain and strategy
    candidates = strategy.ask(1)

    # Add this diagnostic block
    constraint = domain.constraints[0]  # Your NonlinearEqualityConstraint
    constraint_values = constraint(candidates)
    print(f"Constraint value: {constraint_values.iloc[0]}")
    print(f"Constraint value (high precision): {constraint_values.iloc[0]:.20f}")
    print(f"Tolerance: {0.001}")
    print(f"np.isclose result: {np.isclose(constraint_values.iloc[0], 0, atol=0.001)}")

    # Verify constraints are satisfied with relaxed tolerance
    assert len(candidates) == 5
    for _, row in candidates.iterrows():
        constraint_value = abs(row["x1"] + row["x2"] - 0.7)
        assert constraint_value < 1e-3, f"Constraint violated: {constraint_value}"
