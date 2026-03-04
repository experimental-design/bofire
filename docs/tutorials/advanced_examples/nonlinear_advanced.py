"""
Advanced examples of nonlinear constraints in BoFire.
Run this script to verify examples before converting to tutorial format.
"""

import numpy as np
import pandas as pd

import bofire.strategies.api as strategies
from bofire.data_models.constraints.api import (
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.api import MoboStrategy, SoboStrategy


def generate_ring_samples(n_samples=20, r_inner=1.0, r_outer=3.0):
    """Generate samples in a ring (annulus)."""
    samples = []
    for _ in range(n_samples):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(r_inner * 1.05, r_outer * 0.95)
        x1 = r * np.cos(angle)
        x2 = r * np.sin(angle)
        samples.append({"x_1": x1, "x_2": x2})
    return pd.DataFrame(samples)


def example1_nonconvex_ring():
    """Example 1: Nonconvex feasible region (ring)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Nonconvex Ring (1 <= x_1^2 + x_2^2 <= 9)")
    print("=" * 60)

    inputs = [
        ContinuousInput(key="x_1", bounds=(-5, 5)),
        ContinuousInput(key="x_2", bounds=(-5, 5)),
    ]

    outputs = [
        ContinuousOutput(key="y1", objective=MinimizeObjective()),
        ContinuousOutput(key="y2", objective=MaximizeObjective()),
    ]

    # Ring constraint: 1 <= x_1^2 + x_2^2 <= 9
    constraints = [
        NonlinearInequalityConstraint(
            expression="x_1**2 + x_2**2 - 9",  # outer: r² <= 9
            features=["x_1", "x_2"],
        ),
        NonlinearInequalityConstraint(
            expression="1 - x_1**2 - x_2**2",  # inner: r² >= 1 (flipped)
            features=["x_1", "x_2"],
        ),
    ]

    domain = Domain(
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
    )

    # Generate ring samples
    initial_samples = generate_ring_samples(n_samples=20, r_inner=1.0, r_outer=3.0)
    print(f"\nInitial samples (first 5):\n{initial_samples.head()}")

    # Verify ring constraint
    radii = np.sqrt(initial_samples["x_1"] ** 2 + initial_samples["x_2"] ** 2)
    print(f"\nInitial radii: min={radii.min():.3f}, max={radii.max():.3f}")
    print("Expected: [1.0, 3.0]")

    # Mock experiment function
    def mock_experiment(X):
        y1 = X["x_1"] ** 2 + X["x_2"] ** 2
        y2 = -((X["x_1"] - 2) ** 2 + (X["x_2"] - 2) ** 2)
        return pd.DataFrame({"y1": y1, "y2": y2})

    experiments = pd.concat([initial_samples, mock_experiment(initial_samples)], axis=1)

    # Run multi-objective optimization
    strategy_data = MoboStrategy(domain=domain)
    strategy = strategies.map(strategy_data)
    strategy.tell(experiments)

    candidates = strategy.ask(10)
    print(f"\nProposed candidates (first 5):\n{candidates.head()}")

    # Verify constraints
    candidate_radii = np.sqrt(candidates["x_1"] ** 2 + candidates["x_2"] ** 2)
    print(
        f"\nCandidate radii: min={candidate_radii.min():.3f}, max={candidate_radii.max():.3f}"
    )

    for i, constraint in enumerate(domain.constraints, 1):
        fulfilled = constraint.is_fulfilled(candidates, tol=1e-3)
        print(f"Constraint {i} satisfied: {fulfilled.sum()} / {len(candidates)}")


def example2_physics_constraints():
    """Example 2: Physics-based constraints (mass and energy balance)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Physics-Based Constraints (Chemical Process)")
    print("=" * 60)

    inputs = [
        ContinuousInput(key="flow_A", bounds=(0, 10)),  # kg/h
        ContinuousInput(key="flow_B", bounds=(0, 10)),  # kg/h
        ContinuousInput(key="temp", bounds=(300, 400)),  # K
    ]

    outputs = [
        ContinuousOutput(key="yield", objective=MaximizeObjective()),
        ContinuousOutput(key="cost", objective=MinimizeObjective()),
    ]

    # Constraints:
    # 1. Mass balance: flow_A + flow_B = 10
    # 2. Energy balance: flow_A * temp + flow_B * (temp - 50) <= 3500
    constraints = [
        NonlinearEqualityConstraint(
            expression="flow_A + flow_B - 10",
            features=["flow_A", "flow_B"],
        ),
        NonlinearInequalityConstraint(
            expression="flow_A * temp + flow_B * (temp - 50) - 3500",
            features=["flow_A", "flow_B", "temp"],
        ),
    ]

    domain = Domain(
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
    )

    # Generate feasible samples
    samples = []
    for _ in range(20):
        flow_A = np.random.uniform(0, 10)
        flow_B = 10 - flow_A  # Mass balance
        temp = np.random.uniform(300, 350)  # Conservative temp

        # Check energy balance
        energy = flow_A * temp + flow_B * (temp - 50)
        if energy <= 3500:
            samples.append({"flow_A": flow_A, "flow_B": flow_B, "temp": temp})

    initial_samples = pd.DataFrame(samples[:15])
    print(f"\nInitial samples (first 5):\n{initial_samples.head()}")

    # Mock experiment function
    def chemical_experiment(X):
        yield_val = 0.5 * X["flow_A"] + 0.3 * X["flow_B"] + 0.01 * X["temp"]
        cost = 2 * X["flow_A"] + 3 * X["flow_B"] + 0.05 * X["temp"]
        return pd.DataFrame({"yield": yield_val, "cost": cost})

    experiments = pd.concat(
        [initial_samples, chemical_experiment(initial_samples)], axis=1
    )

    strategy_data = MoboStrategy(domain=domain)
    strategy = strategies.map(strategy_data)
    strategy.tell(experiments)

    candidates = strategy.ask(5)
    print(f"\nProposed candidates:\n{candidates}")

    # Verify constraints
    print("\nConstraint verification:")

    # Mass balance
    mass_sum = candidates["flow_A"] + candidates["flow_B"]
    print("\n  Mass balance (should be ~10):")
    print(f"    Min: {mass_sum.min():.6f}")
    print(f"    Max: {mass_sum.max():.6f}")

    # Energy balance
    energy = candidates["flow_A"] * candidates["temp"] + candidates["flow_B"] * (
        candidates["temp"] - 50
    )
    print("\n  Energy balance (should be ≤ 3500):")
    print(f"    Min: {energy.min():.2f}")
    print(f"    Max: {energy.max():.2f}")


def example3_constraint_analysis():
    """Example 3: Detailed constraint violation analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Constraint Violation Analysis")
    print("=" * 60)

    inputs = [
        ContinuousInput(key="x_1", bounds=(-5, 5)),
        ContinuousInput(key="x_2", bounds=(-5, 5)),
        ContinuousInput(key="x_3", bounds=(-5, 5)),
    ]

    outputs = [ContinuousOutput(key="y", objective=MinimizeObjective())]

    # Sphere and plane constraints
    constraints = [
        NonlinearInequalityConstraint(
            expression="x_1**2 + x_2**2 + x_3**2 - 16",  # sphere: r² <= 16
            features=["x_1", "x_2", "x_3"],
        ),
        NonlinearEqualityConstraint(
            expression="x_1 + x_2 + x_3 - 3",  # plane: sum = 3
            features=["x_1", "x_2", "x_3"],
        ),
    ]

    domain = Domain(
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
    )

    # Generate samples on plane within sphere
    samples = []
    for _ in range(25):
        x1 = np.random.uniform(-2, 4)
        x2 = np.random.uniform(-2, 4)
        x3 = 3 - x1 - x2  # On plane

        if x1**2 + x2**2 + x3**2 <= 15.5 and -5 <= x3 <= 5:
            samples.append({"x_1": x1, "x_2": x2, "x_3": x3})

    initial_samples = pd.DataFrame(samples[:20])
    print(f"\nInitial samples (first 5):\n{initial_samples.head()}")

    # Mock function
    def mock_function(X):
        return pd.DataFrame({"y": X["x_1"] ** 2 + X["x_2"] ** 2 + X["x_3"] ** 2})

    experiments = pd.concat([initial_samples, mock_function(initial_samples)], axis=1)

    strategy_data = SoboStrategy(domain=domain)
    strategy = strategies.map(strategy_data)
    strategy.tell(experiments)

    candidates = strategy.ask(10)
    print(f"\nProposed candidates (first 5):\n{candidates.head()}")

    # Detailed analysis
    print("\n" + "=" * 60)
    print("DETAILED CONSTRAINT ANALYSIS")
    print("=" * 60)

    for i, constraint in enumerate(domain.constraints, 1):
        fulfilled = constraint.is_fulfilled(candidates, tol=1e-3)
        violations = constraint(candidates)

        print(f"\nConstraint {i}: {constraint.expression}")
        print(f"  Type: {constraint.__class__.__name__}")
        print(f"  Satisfied: {fulfilled.sum()} / {len(candidates)}")
        print(
            f"  Violations - Min: {violations.min():.6f}, "
            f"Max: {violations.max():.6f}, Mean: {violations.mean():.6f}"
        )


def example4_high_dimensional():
    """Example 4: High-dimensional L2 norm constraint."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: High-Dimensional Constraint (5D L2 Ball)")
    print("=" * 60)

    n_dims = 5
    inputs = [ContinuousInput(key=f"x_{i}", bounds=(-2, 2)) for i in range(n_dims)]

    outputs = [ContinuousOutput(key="y", objective=MinimizeObjective())]

    # L2 ball: sum(x_i²) <= 4
    expr = " + ".join([f"x_{i}**2" for i in range(n_dims)]) + " - 4"

    constraints = [
        NonlinearInequalityConstraint(
            expression=expr,
            features=[f"x_{i}" for i in range(n_dims)],
        )
    ]

    domain = Domain(
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
    )

    # Generate samples in L2 ball
    samples = []
    for _ in range(30):
        # Random direction
        direction = np.random.randn(n_dims)
        direction = direction / np.linalg.norm(direction)

        # Random radius
        r = np.random.uniform(0, 1.9)

        point = r * direction
        sample = {f"x_{i}": point[i] for i in range(n_dims)}
        samples.append(sample)

    initial_samples = pd.DataFrame(samples[:25])
    print(f"\nInitial samples (first 3):\n{initial_samples.head(3)}")

    # Verify L2 norms
    initial_norms = np.sqrt(sum(initial_samples[f"x_{i}"] ** 2 for i in range(n_dims)))
    print(
        f"\nInitial L2 norms: min={initial_norms.min():.3f}, max={initial_norms.max():.3f}"
    )

    # Mock function
    def mock_function(X):
        return pd.DataFrame({"y": sum(X[f"x_{i}"] ** 2 for i in range(n_dims))})

    experiments = pd.concat([initial_samples, mock_function(initial_samples)], axis=1)

    strategy_data = SoboStrategy(domain=domain)
    strategy = strategies.map(strategy_data)
    strategy.tell(experiments)

    candidates = strategy.ask(5)
    print(f"\nProposed candidates (first 3):\n{candidates.head(3)}")

    # Check L2 norms
    candidate_norms = np.sqrt(sum(candidates[f"x_{i}"] ** 2 for i in range(n_dims)))
    print("\nCandidate L2 norms:")
    print(f"  Min: {candidate_norms.min():.3f}")
    print(f"  Max: {candidate_norms.max():.3f}")
    print("  Expected: ≤ 2.0")

    # Constraint check
    for constraint in domain.constraints:
        fulfilled = constraint.is_fulfilled(candidates, tol=1e-3)
        print(f"\nConstraint satisfied: {fulfilled.sum()} / {len(candidates)}")


if __name__ == "__main__":
    example1_nonconvex_ring()
    example2_physics_constraints()
    example3_constraint_analysis()
    example4_high_dimensional()

    print("\n" + "=" * 60)
    print("ALL ADVANCED EXAMPLES COMPLETED!")
    print("=" * 60)
