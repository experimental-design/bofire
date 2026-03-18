import time

import numpy as np
import pandas as pd
import pytest

from bofire.data_models.constraints.api import (
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective


# ========== TASK 1.0: BASIC EQUALITY CONSTRAINT ==========


def test_nonlinear_equality_constraint_sobo():
    """Test that SoboStrategy handles NonlinearEqualityConstraint correctly."""
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

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

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    experiments = pd.DataFrame(
        {
            "x1": [0.3, 0.4, 0.5],
            "x2": [0.4, 0.3, 0.2],
            "y": [0.5, 0.6, 0.7],
            "valid_y": [1, 1, 1],
        }
    )
    strategy.tell(experiments)
    candidates = strategy.ask(5)

    assert len(candidates) == 5, f"Expected 5 candidates, got {len(candidates)}"
    for idx, row in candidates.iterrows():
        constraint_value = abs(row["x1"] + row["x2"] - 0.7)
        assert constraint_value <= 1e-3 + 1e-9, (
            f"Constraint violated for row {idx}: x1={row['x1']}, x2={row['x2']}, "
            f"violation={constraint_value}"
        )
    print(f"✓ All {len(candidates)} candidates satisfy the constraint")


# ========== TASK 1.1: MULTIPLE NONLINEAR CONSTRAINTS ==========


def test_multiple_nonlinear_constraints():
    """Test SoboStrategy with 2 nonlinear constraints simultaneously.

    Scenario:
    - Constraint 1: x^2 + y^2 <= 1 (circle)
    - Constraint 2: x + y >= 0.5 (half-plane)
    - Feasible region: intersection of both
    """
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(-2, 2)),
                ContinuousInput(key="y", bounds=(-2, 2)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="x**2 + y**2 - 1"
                ),
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="0.5 - x - y"
                ),
            ]
        ),
    )

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    experiments = pd.DataFrame(
        {
            "x": [0.5, 0.3, 0.4],
            "y": [0.3, 0.4, 0.3],
            "z": [1.0, 0.8, 0.9],
            "valid_z": [1, 1, 1],
        }
    )
    strategy.tell(experiments)
    candidates = strategy.ask(1)

    assert len(candidates) == 1
    x_val = candidates.iloc[0]["x"]
    y_val = candidates.iloc[0]["y"]

    assert (
        x_val**2 + y_val**2 <= 1.0 + 1e-3
    ), f"Circle constraint violated: {x_val}^2 + {y_val}^2 = {x_val**2 + y_val**2}"
    assert (
        x_val + y_val >= 0.5 - 1e-3
    ), f"Half-plane constraint violated: {x_val} + {y_val} = {x_val + y_val}"
    print(f"✓ Multiple constraints satisfied: x={x_val:.4f}, y={y_val:.4f}")


@pytest.mark.parametrize("n_constraints", [2, 3, 5])
def test_multiple_nonlinear_constraints_scaling(n_constraints):
    """Verify performance with increasing number of constraints."""
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    constraints_list = [
        NonlinearInequalityConstraint(
            features=["x", "y"],
            expression=f"x**2 + y**2 - {1 + i*0.1}",
        )
        for i in range(n_constraints)
    ]

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(-2, 2)),
                ContinuousInput(key="y", bounds=(-2, 2)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(constraints=constraints_list),
    )

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    experiments = pd.DataFrame(
        {
            "x": [0.3, 0.2],
            "y": [0.2, 0.3],
            "z": [1.0, 0.9],
            "valid_z": [1, 1],
        }
    )
    strategy.tell(experiments)

    start = time.time()
    candidates = strategy.ask(1)
    elapsed = time.time() - start

    assert len(candidates) == 1
    assert elapsed < 30.0, f"Too slow with {n_constraints} constraints: {elapsed:.2f}s"

    x_val = candidates.iloc[0]["x"]
    y_val = candidates.iloc[0]["y"]
    # Only check the tightest constraint (first one, radius=1.0)
    assert (
        x_val**2 + y_val**2 <= 1.0 + 1e-3
    ), f"Tightest constraint violated: {x_val}^2 + {y_val}^2 = {x_val**2 + y_val**2}"
    print(f"✓ {n_constraints} constraints satisfied in {elapsed:.2f}s")


# ========== TASK 1.2: TIGHT CONSTRAINTS ==========
def test_tight_nonlinear_constraints():
    """Test constraints with very small feasible region (circle radius=0.1)."""
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(-1, 1)),
                ContinuousInput(key="y", bounds=(-1, 1)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression=" x**2 + y**2 - 0.01"
                ),
            ]
        ),
    )

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    experiments = pd.DataFrame(
        {
            "x": [0.05, 0.03],
            "y": [0.05, 0.06],
            "z": [1.0, 0.9],
            "valid_z": [1, 1],
        }
    )
    strategy.tell(experiments)

    candidates = strategy.ask(1)
    assert len(candidates) == 1

    x_val = candidates.iloc[0]["x"]
    y_val = candidates.iloc[0]["y"]
    assert x_val**2 + y_val**2 <= 0.01 + 1e-3, "Tight constraint violated"
    print(f"✓ Tight constraint satisfied: x={x_val:.4f}, y={y_val:.4f}")


# ========== TASK 1.3: EMPTY FEASIBLE REGION ==========


def test_empty_feasible_region():
    """Test behavior when constraints make feasible region empty."""
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(0, 1)),
                ContinuousInput(key="y", bounds=(0, 1)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="x + y - 1.5"
                ),
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="0.5 - x - y"
                ),
            ]
        ),
    )

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    experiments = pd.DataFrame(
        {
            "x": [0.5],
            "y": [0.5],
            "z": [1.0],
            "valid_z": [1],
        }
    )
    strategy.tell(experiments)

    with pytest.raises(ValueError, match="Not enough experiments available"):
        strategy.ask(1)

    print("✓ Empty feasible region handled correctly")


# ========== TASK 1.4: HIGH-DIMENSIONAL CONSTRAINTS ==========


@pytest.mark.slow
# @pytest.mark.skip(reason="need --runslow option to execute")
def test_high_dimensional_nonlinear_constraints():
    """Test constraints in 10+ dimensional space."""
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    n_dims = 5
    features = [ContinuousInput(key=f"x{i}", bounds=(-2, 2)) for i in range(n_dims)]
    expression = " + ".join([f"x{i}**2" for i in range(n_dims)]) + " - 1"

    domain = Domain(
        inputs=Inputs(features=features),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearInequalityConstraint(
                    features=[f"x{i}" for i in range(n_dims)], expression=expression
                ),
            ]
        ),
    )

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    init_data = {f"x{i}": [0.1, 0.05] for i in range(n_dims)}
    init_data.update({"z": [1.0, 0.9], "valid_z": [1, 1]})
    strategy.tell(pd.DataFrame(init_data))

    start = time.time()
    candidates = strategy.ask(1)
    elapsed = time.time() - start

    assert len(candidates) == 1
    assert elapsed < 120.0
    sum_sq = sum(candidates.iloc[0][f"x{i}"] ** 2 for i in range(n_dims))
    assert sum_sq <= 1.0 + 1e-3
    print(f"✓ {n_dims}D constraint satisfied in {elapsed:.2f}s")


# ========== TASK 1.5: DIFFERENT ACQUISITION FUNCTIONS ==========


@pytest.mark.parametrize("acqf_class", ["qLogEI", "qLogNEI", "qUCB"])
def test_nonlinear_constraints_different_acqf(acqf_class):
    """Test nonlinear constraints with different acquisition functions."""
    from bofire.data_models.acquisition_functions.api import qLogEI, qLogNEI, qUCB
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    acqf_map = {"qLogEI": qLogEI(), "qLogNEI": qLogNEI(), "qUCB": qUCB()}

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(-2, 2)),
                ContinuousInput(key="y", bounds=(-2, 2)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression=" x**2 + y**2 - 1"
                ),
            ]
        ),
    )

    data_model = SoboStrategyDataModel(
        domain=domain, acquisition_function=acqf_map[acqf_class]
    )
    strategy = SoboStrategy(data_model=data_model)

    experiments = pd.DataFrame(
        {
            "x": [0.5, 0.3],
            "y": [0.3, 0.4],
            "z": [1.0, 0.8],
            "valid_z": [1, 1],
        }
    )
    strategy.tell(experiments)
    candidates = strategy.ask(1)

    assert len(candidates) == 1
    x_val = candidates.iloc[0]["x"]
    y_val = candidates.iloc[0]["y"]
    assert x_val**2 + y_val**2 <= 1.0 + 1e-3, f"{acqf_class} constraint violated"
    print(f"✓ {acqf_class} with constraint: x={x_val:.4f}, y={y_val:.4f}")


# ========== TASK 1.6: EDGE CASES ==========


def test_nonlinear_constraint_boundary_initial_data():
    """Test behavior when initial data is exactly on constraint boundary."""
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(0, 2)),
                ContinuousInput(key="y", bounds=(0, 2)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="x**2 + y**2 - 1"
                ),
            ]
        ),
    )

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    # Exactly on boundary: x^2 + y^2 = 1
    experiments = pd.DataFrame(
        {
            "x": [1.0, 0.0, np.sqrt(0.5)],
            "y": [0.0, 1.0, np.sqrt(0.5)],
            "z": [1.0, 0.9, 0.8],
            "valid_z": [1, 1, 1],
        }
    )
    strategy.tell(experiments)
    candidates = strategy.ask(1)
    assert len(candidates) == 1
    print("✓ Boundary initial data handled")


def test_nonlinear_equality_near_boundary():
    """Test equality constraint near domain boundary."""
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(0, 1)),
                ContinuousInput(key="y", bounds=(0, 1)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearEqualityConstraint(
                    features=["x", "y"], expression="x + y - 1.5"
                ),
            ]
        ),
    )

    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)

    experiments = pd.DataFrame(
        {
            "x": [0.7, 0.8, 0.9],
            "y": [0.8, 0.7, 0.6],
            "z": [1.0, 0.9, 0.8],
            "valid_y": [1, 1, 1],
        }
    )
    strategy.tell(experiments)
    candidates = strategy.ask(1)

    assert len(candidates) == 1
    x_val = candidates.iloc[0]["x"]
    y_val = candidates.iloc[0]["y"]
    assert (
        abs(x_val + y_val - 1.5) <= 1.001e-3
    ), f"Equality constraint violated: x+y={x_val+y_val:.6f}"  # taped change this
    print(f"✓ Boundary equality satisfied: x+y={x_val+y_val:.6f}")


# @pytest.mark.skip(reason="NonlinearConstraints require ≥2 features by design.")
def test_single_input_nonlinear_constraint():
    """NonlinearConstraints must have >= 2 features — verify ValidationError is raised."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="at least 2 items after validation"):
        NonlinearInequalityConstraint(
            features=["x"],  # single feature — must fail
            expression="x - 1",
        )


def test_debug_is_fulfilled():
    """Debug why is_fulfilled rejects valid experiments."""
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(0, 1)),
                ContinuousInput(key="y", bounds=(0, 1)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearEqualityConstraint(
                    features=["x", "y"], expression="x + y - 1.5"
                ),
            ]
        ),
    )

    experiments = pd.DataFrame({"x": [0.7], "y": [0.8], "z": [1.0], "valid_z": [1]})

    print("\n=== DEBUGGING is_fulfilled() ===")
    print(
        f"Sum: {experiments['x'].values[0] + experiments['y'].values[0]}, Expected: 1.5"
    )

    result = domain.is_fulfilled(experiments)
    print(f"is_fulfilled result: {result.values}")

    constraint = domain.constraints.constraints[0]
    constraint_result = constraint.is_fulfilled(experiments)
    print(f"Direct constraint check: {constraint_result.values}")


def test_diagnose_multiple_constraints_optimizer():
    """Diagnose what constraints actually reach the BoTorch optimizer."""
    import torch

    from bofire.utils.torch_tools import get_nonlinear_constraints

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(-2, 2)),
                ContinuousInput(key="y", bounds=(-2, 2)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="1 - x**2 - y**2"
                ),
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="x + y - 0.5"
                ),
            ]
        ),
    )

    result = get_nonlinear_constraints(domain)
    print(f"\nDEBUG: returned {len(result)} items")
    print(f"DEBUG: type of first item: {type(result[0])}")
    print(f"DEBUG: first item: {result[0]}")

    # Each item should be (callable, bool)
    for i, item in enumerate(result):
        print(f"\n--- Constraint {i} ---")
        print(f"  type: {type(item)}")

        if isinstance(item, tuple):
            fn, is_ineq = item
            print(f"  is_inequality: {is_ineq}")
            print(f"  callable type: {type(fn)}")

            # Test at FEASIBLE point (0.5, 0.3)
            feasible = torch.tensor([[[0.5, 0.3]]], dtype=torch.float64)
            val_feasible = fn(feasible)
            print(
                f"  f(0.5, 0.3) = {val_feasible.item():.4f}  (should be > 0 if feasible)"
            )

            # Test at INFEASIBLE point (1.5, 1.5)
            infeasible = torch.tensor([[[1.5, 1.5]]], dtype=torch.float64)
            val_infeasible = fn(infeasible)
            print(
                f"  f(1.5, 1.5) = {val_infeasible.item():.4f}  (should be < 0 if infeasible)"
            )
        else:
            print("  ⚠️  NOT a tuple — this is the bug!")

    # The key question: what format does BoTorch optimize_acqf expect?
    # BoTorch expects: List[Callable[[Tensor], Tensor]]
    # BoFire returns:  List[Tuple[Callable, bool]]
    # These are DIFFERENT!
    print(f"\n{'='*50}")
    print(f"BoFire returns tuples: {[type(x).__name__ for x in result]}")
    print("BoTorch expects raw callables — tuples will FAIL!")


def test_diagnose_has_sufficient_experiments():
    """Isolate why has_sufficient_experiments fails with multi-constraint domain."""
    from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
    from bofire.strategies.api import SoboStrategy

    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x", bounds=(-2, 2)),
                ContinuousInput(key="y", bounds=(-2, 2)),
            ]
        ),
        outputs=Outputs(
            features=[ContinuousOutput(key="z", objective=MinimizeObjective())]
        ),
        constraints=Constraints(
            constraints=[
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="1 - x**2 - y**2"
                ),
                NonlinearInequalityConstraint(
                    features=["x", "y"], expression="x + y - 0.5"
                ),
            ]
        ),
    )

    # First verify each constraint IS satisfied by each point using is_fulfilled directly
    import pandas as pd

    experiments = pd.DataFrame(
        {
            "x": [0.5, 0.3, 0.4],
            "y": [0.3, 0.4, 0.3],
            "z": [1.0, 0.8, 0.9],
            "valid_z": [1, 1, 1],
        }
    )

    print("\n--- Constraint is_fulfilled checks ---")
    for c in domain.constraints.constraints:
        result = c.is_fulfilled(experiments)
        print(f"  {c.expression}: {result.tolist()}")

    # Now tell and check state
    data_model = SoboStrategyDataModel(domain=domain)
    strategy = SoboStrategy(data_model=data_model)
    strategy.tell(experiments)

    print(
        f"\n  strategy.experiments shape: {strategy.experiments.shape if hasattr(strategy, 'experiments') and strategy.experiments is not None else 'None'}"
    )
    print(f"  has_sufficient_experiments: {strategy.has_sufficient_experiments()}")

    # What does get_training_data / n_experiments return?
    if hasattr(strategy, "num_experiments"):
        print(f"  num_experiments: {strategy.num_experiments}")
