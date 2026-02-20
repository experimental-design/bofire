import pandas as pd

from bofire.data_models.acquisition_functions.api import qEI
from bofire.data_models.constraints.api import NonlinearEqualityConstraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.api import SoboStrategy


# Define domain with equality constraint: x1 + x2 = 0
domain = Domain.from_lists(
    inputs=[
        ContinuousInput(key="x1", bounds=(-1, 1)),
        ContinuousInput(key="x2", bounds=(-1, 1)),
    ],
    outputs=[
        ContinuousOutput(key="y"),
    ],
    constraints=[
        NonlinearEqualityConstraint(
            expression="x1 + x2",  # = 0
            features=["x1", "x2"],
        )
    ],
)

# Initial data
initial_experiments = pd.DataFrame(
    {"x1": [0.0, 0.5, -0.5], "x2": [0.0, -0.5, 0.5], "y": [1.0, 0.8, 0.9]}
)

strategy = SoboStrategy(domain=domain, acquisition_function=qEI())
strategy.tell(initial_experiments)

print("Test: SOBO + NonlinearEquality")
print("=" * 60)
try:
    candidates = strategy.ask(1)
    print("✅ SUCCESS!")
    print(candidates)

    constraint = domain.constraints[0]
    results = constraint(candidates)
    print(f"\n✅ Constraint values: {results.values}")
    print(f"✅ Satisfied? {(abs(results) < 1e-6).all()}")

except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
