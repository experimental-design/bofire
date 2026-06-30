"""Test: Are constraints actually being passed to BoTorch?"""

import pandas as pd

from bofire.data_models.constraints.api import NonlinearInequalityConstraint
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.strategies.api import SoboStrategy


domain = Domain(
    inputs=Inputs(
        features=[
            ContinuousInput(key="x", bounds=(-2, 2)),
            ContinuousInput(key="y", bounds=(-2, 2)),
        ]
    ),
    outputs=Outputs(features=[ContinuousOutput(key="z")]),
    constraints=[
        NonlinearInequalityConstraint(
            features=["x", "y"], expression="1 - x**2 - y**2"
        ),
        NonlinearInequalityConstraint(features=["x", "y"], expression="x + y - 0.5"),
    ],
)

initial_data = pd.DataFrame({"x": [0.3], "y": [0.3], "z": [0.18], "valid_z": [True]})

# Test with default BotorchOptimizer (just to confirm behavior)
data_model = SoboStrategyDataModel(domain=domain)
strategy = SoboStrategy(data_model=data_model)
strategy.tell(initial_data)

print("Testing with default BotorchOptimizer...")
try:
    candidates = strategy.ask(5)
    print(f"✅ Generated {len(candidates)} candidates")

    # Check if they satisfy constraints
    for _, row in candidates.iterrows():
        x, y = row["x"], row["y"]
        c1 = 1 - x**2 - y**2  # Should be >= 0
        c2 = x + y - 0.5  # Should be >= 0
        valid = c1 >= -1e-3 and c2 >= -1e-3
        print(f"  x={x:.3f}, y={y:.3f} | c1={c1:.4f}, c2={c2:.4f} | Valid={valid}")

except Exception as e:
    print(f"❌ FAILED: {e}")
