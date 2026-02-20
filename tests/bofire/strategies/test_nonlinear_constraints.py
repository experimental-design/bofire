import pandas as pd

from bofire.data_models.constraints.api import NonlinearEqualityConstraint
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective


def test_nonlinear_equality_constraint_sobo():
    """Test that SoboStrategy handles NonlinearEqualityConstraint correctly."""

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

    # Ask for 5 candidates (matching the assertion below)
    candidates = strategy.ask(5)

    # Verify we got the expected number of candidates
    assert len(candidates) == 5, f"Expected 5 candidates, got {len(candidates)}"

    # Verify all constraints are satisfied
    # constraint = domain.constraints[0]
    for idx, row in candidates.iterrows():
        constraint_value = abs(row["x1"] + row["x2"] - 0.7)
        # Use slightly relaxed tolerance to account for floating-point precision
        assert constraint_value <= 1e-3 + 1e-9, (
            f"Constraint violated for row {idx}: "
            f"x1={row['x1']}, x2={row['x2']}, "
            f"violation={constraint_value}"
        )

    print(f"✓ All {len(candidates)} candidates satisfy the constraint")
