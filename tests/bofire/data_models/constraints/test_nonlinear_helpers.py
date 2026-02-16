import pandas as pd
from bofire.data_models.constraints.api import (
    NonlinearInequalityConstraint,
    filter_candidates_by_constraints,
    get_constraint_violations,
)

print("✅ Imports successful!")

# Create a simple constraint: x^2 + y^2 <= 1 (unit circle)
constraint = NonlinearInequalityConstraint(
    features=["x", "y"],
    expression="x**2 + y**2 - 1"
)

# Test data: 3 points (2 inside circle, 1 outside)
candidates = pd.DataFrame({
    "x": [0.5, 1.5, 0.0],
    "y": [0.5, 0.5, 0.0],
})

print("\n🔍 Test 1: Constraint evaluation")
print(constraint(candidates))

print("\n🔍 Test 2: Filter feasible candidates")
feasible = filter_candidates_by_constraints(candidates, [constraint])
print(feasible)
print(f"Found {len(feasible)} feasible candidates (expected 2)")

print("\n🔍 Test 3: Get constraint violations")
violations = get_constraint_violations(candidates, [constraint])
print(violations)

print("\n🎉 All tests passed!")
