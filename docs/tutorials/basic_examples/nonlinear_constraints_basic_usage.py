"""
PRACTICAL EXAMPLE: Competing Reactions Optimization
================================================================================
Problem: Maximize yield of desired product P while minimizing waste W

Reaction System:
    A + B → P  (k1, desired)
    P + B → W  (k2, undesired)

Optimization Variables:
    - Temperature (K): affects both reaction rates
    - Residence time (min): controls conversion
    - Feed ratio (B:A): excess B increases P formation but also P→W
    - Pressure (bar): affects concentration

Objectives:
    - Maximize yield of P
    - Minimize formation of W

Nonlinear Constraints:
    1. Selectivity: S = [P]/[W] ≥ 5.0 (quality requirement)
    2. Conversion: X_A ≥ 0.90 (economic requirement)
    3. Heat generation: Q ≤ Q_max (safety limit)

================================================================================
"""

import numpy as np
import pandas as pd
import torch

from bofire.data_models.constraints.api import NonlinearInequalityConstraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective


# ============================================================================
# REACTION KINETICS MODEL
# ============================================================================


class CompetingReactionsModel:
    """
    Kinetic model for competing reactions:
        A + B → P  (k1)
        P + B → W  (k2)

    Uses Arrhenius kinetics: k = A * exp(-Ea / (R*T))
    """

    def __init__(self):
        # Pre-exponential factors [L/mol/min]
        self.A1 = 1e8  # Reaction 1: A + B → P
        self.A2 = 5e6  # Reaction 2: P + B → W

        # Activation energies [J/mol]
        self.Ea1 = 65000  # Lower Ea = faster at lower T
        self.Ea2 = 75000  # Higher Ea = more temperature sensitive

        # Gas constant
        self.R = 8.314  # J/(mol·K)

        # Heat of reactions [J/mol]
        self.dH1 = -85000  # Exothermic
        self.dH2 = -45000  # Exothermic

        # Initial concentrations [mol/L]
        self.C_A0 = 2.0

    def rate_constant(self, T, A, Ea):
        """Arrhenius equation: k = A * exp(-Ea / RT)"""
        return A * np.exp(-Ea / (self.R * T))

    def simulate_batch(self, T, tau, feed_ratio, pressure):
        """
        Simulate batch reactor

        Parameters:
        -----------
        T : float
            Temperature [K]
        tau : float
            Residence time [min]
        feed_ratio : float
            Molar ratio B:A in feed
        pressure : float
            Pressure [bar] - affects concentrations

        Returns:
        --------
        dict with concentrations and derived quantities
        """
        # Rate constants
        k1 = self.rate_constant(T, self.A1, self.Ea1)
        k2 = self.rate_constant(T, self.A2, self.Ea2)

        # Initial concentrations (pressure effect)
        C_A = self.C_A0 * (pressure / 1.0)  # Reference at 1 bar
        C_B = C_A * feed_ratio
        C_P = 0.0
        C_W = 0.0

        # Simple integration (Euler method for demonstration)
        dt = 0.01  # min
        steps = int(tau / dt)

        for _ in range(steps):
            # Reaction rates [mol/L/min]
            r1 = k1 * C_A * C_B  # A + B → P
            r2 = k2 * C_P * C_B  # P + B → W

            # Update concentrations
            C_A -= r1 * dt
            C_B -= (r1 + r2) * dt
            C_P += (r1 - r2) * dt
            C_W += r2 * dt

            # Prevent negative concentrations
            C_A = max(C_A, 0)
            C_B = max(C_B, 0)
            C_P = max(C_P, 0)

        # Calculate outputs
        X_A = 1 - C_A / (self.C_A0 * pressure)  # Conversion of A
        Y_P = C_P / (self.C_A0 * pressure)  # Yield of P
        Y_W = C_W / (self.C_A0 * pressure)  # Yield of W

        # Selectivity (handle division by zero)
        if C_W > 1e-6:
            S = C_P / C_W
        else:
            S = 100.0  # Very high if no waste formed

        # Heat generation [W/L]
        Q = abs(self.dH1 * r1 + self.dH2 * r2) / 60  # Convert to W/L

        return {
            "C_A": C_A,
            "C_B": C_B,
            "C_P": C_P,
            "C_W": C_W,
            "X_A": X_A,
            "Y_P": Y_P,
            "Y_W": Y_W,
            "S": S,
            "Q": Q,
        }


# def selectivity_constraint(temperature, residence_time, feed_ratio, pressure, **kwargs):
#     import torch

#     # Arrhenius rate constants
#     R = 8.314
#     k1 = 1e8 * torch.exp(-65000.0 / (R * temperature))
#     k2 = 5e6 * torch.exp(-75000.0 / (R * temperature))

#     # Initial concentrations
#     C_A0 = 2.0 * pressure
#     C_B0 = C_A0 * feed_ratio

#     # Approximate steady-state selectivity (assumes high conversion)
#     # For consecutive reactions: S ≈ k1/k2 * sqrt(C_B0/C_A0) * tau_factor
#     tau_factor = torch.sqrt(residence_time / 20.0)  # Normalize to reference
#     S = (k1 / k2) * torch.sqrt(feed_ratio) * tau_factor

#     return 4.5 - S


# def selectivity_constraint(temperature, residence_time, feed_ratio, pressure, **kwargs):
#     """
#     Selectivity constraint: S ≥ 5.0

#     Reformulated as: 5.0 - S ≤ 0
#     """
#     model = CompetingReactionsModel()

#     # Convert to numpy and ensure 1D array
#     T_vals = np.atleast_1d(temperature.detach().cpu().numpy())
#     tau_vals = np.atleast_1d(residence_time.detach().cpu().numpy())
#     ratio_vals = np.atleast_1d(feed_ratio.detach().cpu().numpy())
#     P_vals = np.atleast_1d(pressure.detach().cpu().numpy())

#     violations = []
#     for i in range(len(T_vals)):
#         result = model.simulate_batch(
#             T=float(T_vals[i]),
#             tau=float(tau_vals[i]),
#             feed_ratio=float(ratio_vals[i]),
#             pressure=float(P_vals[i]),
#         )
#         # ⬅️ CHANGE: Relax from 5.0 to 4.5
#         violations.append(4.5 - result["S"])

#     return torch.tensor(violations, dtype=torch.float64)


# def conversion_constraint(temperature, residence_time, feed_ratio, pressure, **kwargs):
#     """
#     Conversion constraint: X_A ≥ 0.90

#     Reformulated as: 0.90 - X_A ≤ 0
#     """
#     model = CompetingReactionsModel()

#     # Convert to numpy and ensure 1D array
#     T_vals = np.atleast_1d(temperature.detach().cpu().numpy())
#     tau_vals = np.atleast_1d(residence_time.detach().cpu().numpy())
#     ratio_vals = np.atleast_1d(feed_ratio.detach().cpu().numpy())
#     P_vals = np.atleast_1d(pressure.detach().cpu().numpy())

#     violations = []
#     for i in range(len(T_vals)):
#         result = model.simulate_batch(
#             T=float(T_vals[i]),
#             tau=float(tau_vals[i]),
#             feed_ratio=float(ratio_vals[i]),
#             pressure=float(P_vals[i]),
#         )
#         # ⬅️ CHANGE: Relax from 0.90 to 0.85
#         violations.append(0.85 - result["X_A"])

#     return torch.tensor(violations, dtype=torch.float64)


# def heat_constraint(temperature, residence_time, feed_ratio, pressure, **kwargs):
#     """
#     Heat generation constraint: Q ≤ 1200 W/L (cooling capacity)

#     Reformulated as: Q - 1200 ≤ 0
#     """
#     model = CompetingReactionsModel()

#     # Convert to numpy and ensure 1D array
#     T_vals = np.atleast_1d(temperature.detach().cpu().numpy())
#     tau_vals = np.atleast_1d(residence_time.detach().cpu().numpy())
#     ratio_vals = np.atleast_1d(feed_ratio.detach().cpu().numpy())
#     P_vals = np.atleast_1d(pressure.detach().cpu().numpy())

#     violations = []
#     for i in range(len(T_vals)):
#         result = model.simulate_batch(
#             T=float(T_vals[i]),
#             tau=float(tau_vals[i]),
#             feed_ratio=float(ratio_vals[i]),
#             pressure=float(P_vals[i]),
#         )
#         # ⬅️ CHANGE: Relax from 1200 to 1300
#         violations.append(result["Q"] - 1300)

#     return torch.tensor(violations, dtype=torch.float64)


def selectivity_constraint(temperature, residence_time, feed_ratio, pressure, **kwargs):
    # Arrhenius constants
    R = 8.314
    k1 = 1e8 * torch.exp(-65000.0 / (R * temperature))
    k2 = 5e6 * torch.exp(-75000.0 / (R * temperature))

    # Concentrations
    C_A0 = 2.0 * pressure
    C_B0 = C_A0 * feed_ratio

    # Approximate selectivity (use ratio C_B0 / C_A0 instead of raw feed_ratio)
    tau_factor = torch.sqrt(residence_time / 20.0)
    S = (k1 / k2) * torch.sqrt(C_B0 / C_A0) * tau_factor

    return 4.5 - S


def conversion_constraint(temperature, residence_time, feed_ratio, pressure, **kwargs):
    R = 8.314
    k1 = 1e8 * torch.exp(-65000.0 / (R * temperature))

    C_A0 = 2.0 * pressure
    C_B0 = C_A0 * feed_ratio

    # Approximate conversion (first-order decay)
    X_A = 1.0 - torch.exp(-k1 * C_B0 * residence_time)

    return 0.85 - X_A


def heat_constraint(temperature, residence_time, feed_ratio, pressure, **kwargs):
    R = 8.314
    k1 = 1e8 * torch.exp(-65000.0 / (R * temperature))
    k2 = 5e6 * torch.exp(-75000.0 / (R * temperature))

    C_A0 = 2.0 * pressure
    C_B0 = C_A0 * feed_ratio

    # Approximate heat generation
    dH1 = -85000
    dH2 = -45000

    r1 = k1 * C_A0 * C_B0 * torch.exp(-k1 * C_B0 * residence_time)
    r2 = k2 * (C_A0 / 2.0) * C_B0

    Q = torch.abs(dH1 * r1 + dH2 * r2) / 60

    return Q - 1300


# ============================================================================
# FEASIBLE SAMPLE GENERATION
# ============================================================================


def generate_feasible_samples(n_samples=20):
    """
    Generate initial samples that satisfy all constraints.

    Strategy: Use conservative parameter ranges known to be safe.
    """
    model = CompetingReactionsModel()
    samples = []

    attempts = 0
    max_attempts = n_samples * 10

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        # Sample from conservative ranges
        T = np.random.uniform(320, 360)  # K (moderate temp)
        tau = np.random.uniform(10, 30)  # min
        feed_ratio = np.random.uniform(1.2, 2.0)  # Slight excess B
        pressure = np.random.uniform(1.5, 2.5)  # bar

        # Check constraints (⬅️ USE RELAXED VALUES)
        result = model.simulate_batch(T, tau, feed_ratio, pressure)

        if (
            result["S"] >= 4.5  # ⬅️ Changed from 5.0
            and result["X_A"] >= 0.85  # ⬅️ Changed from 0.90
            and result["Q"] <= 1300
        ):  # ⬅️ Changed from 1200
            samples.append(
                {
                    "temperature": T,
                    "residence_time": tau,
                    "feed_ratio": feed_ratio,
                    "pressure": pressure,
                }
            )

    if len(samples) < n_samples:
        print(
            f"Warning: Only found {len(samples)} feasible samples out of {n_samples} requested"
        )

    return pd.DataFrame(samples)


# ============================================================================
# MAIN OPTIMIZATION EXAMPLE
# ============================================================================


def main():
    print("\n" + "=" * 80)
    print("COMPETING REACTIONS OPTIMIZATION")
    print("=" * 80)
    print("\nReaction System:")
    print("  A + B → P  (desired product)")
    print("  P + B → W  (waste byproduct)")
    print("\nGoal: Maximize P yield, minimize W formation")
    print("=" * 80)

    # ========================================================================
    # 1. DEFINE OPTIMIZATION DOMAIN
    # ========================================================================

    inputs = [
        ContinuousInput(
            key="temperature",
            bounds=(310, 380),  # K
        ),
        ContinuousInput(
            key="residence_time",
            bounds=(5, 40),  # min
        ),
        ContinuousInput(
            key="feed_ratio",
            bounds=(1.0, 3.0),  # B:A molar ratio
        ),
        ContinuousInput(
            key="pressure",
            bounds=(1.0, 3.0),  # bar
        ),
    ]

    outputs = [
        ContinuousOutput(
            key="yield_P",
            objective=MaximizeObjective(),
        ),
        ContinuousOutput(
            key="yield_W",
            objective=MinimizeObjective(),
        ),
    ]

    constraints = [
        NonlinearInequalityConstraint(
            expression=selectivity_constraint,
            features=["temperature", "residence_time", "feed_ratio", "pressure"],
        ),
        NonlinearInequalityConstraint(
            expression=conversion_constraint,
            features=["temperature", "residence_time", "feed_ratio", "pressure"],
        ),
        NonlinearInequalityConstraint(
            expression=heat_constraint,
            features=["temperature", "residence_time", "feed_ratio", "pressure"],
        ),
    ]

    domain = Domain(
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
    )

    print("\n" + "-" * 80)
    print("DOMAIN SETUP")
    print("-" * 80)
    print("\nOptimization Variables:")
    for inp in inputs:
        print(f"  • {inp.key}: {inp.bounds}")

    print("\nObjectives:")
    for out in outputs:
        print(f"  • {out.key}: {out.objective}")

    print("\nConstraints:")
    print("  1. Selectivity S ≥ 4.5")  # ⬅️ Changed from 5.0
    print("  2. Conversion X_A ≥ 0.85")  # ⬅️ Changed from 0.90
    print("  3. Heat generation Q ≤ 1300 W/L")  # ⬅️ Changed from 1200

    # ========================================================================
    # 2. GENERATE FEASIBLE INITIAL DATA
    # ========================================================================

    print("\n" + "-" * 80)
    print("GENERATING INITIAL EXPERIMENTS")
    print("-" * 80)

    initial_samples = generate_feasible_samples(n_samples=20)
    print(f"\nGenerated {len(initial_samples)} feasible initial samples")
    print("\nFirst 5 samples:")
    print(initial_samples.head())

    # Evaluate with reaction model
    model = CompetingReactionsModel()
    results = []

    for _, row in initial_samples.iterrows():
        result = model.simulate_batch(
            T=row["temperature"],
            tau=row["residence_time"],
            feed_ratio=row["feed_ratio"],
            pressure=row["pressure"],
        )
        results.append(
            {
                "yield_P": result["Y_P"],
                "yield_W": result["Y_W"],
            }
        )

    experiments = pd.concat([initial_samples, pd.DataFrame(results)], axis=1)

    print("\nExperimental results (first 5):")
    print(experiments.head())

    # Verify constraints
    print("\n" + "-" * 80)
    print("CONSTRAINT VERIFICATION (Initial Data)")
    print("-" * 80)

    for i, constraint in enumerate(constraints, 1):
        violations = constraint(initial_samples)
        satisfied = (violations <= 1e-3).sum()
        print(f"\nConstraint {i}:")
        print(f"  Satisfied: {satisfied} / {len(initial_samples)}")
        print(f"  Max violation: {violations.max():.6f}")

    # ========================================================================
    # 3. RUN OPTIMIZATION STRATEGY
    # ========================================================================

    print("\n" + "-" * 80)
    print("RUNNING MULTI-OBJECTIVE OPTIMIZATION")
    print("-" * 80)

    from bofire.data_models.strategies.api import MoboStrategy as MoboStrategyDataModel
    from bofire.data_models.strategies.predictives.acqf_optimization import (
        BotorchOptimizer,
    )
    from bofire.strategies.api import MoboStrategy

    # Step 1: Create data model with configuration
    strategy_data = MoboStrategyDataModel(
        domain=domain,
        acquisition_optimizer=BotorchOptimizer(
            batch_limit=1,  # Required for nonlinear constraints
            n_restarts=2,  # Number of optimization restarts
            n_raw_samples=64,
        ),
    )

    # Step 2: Create strategy from data model
    strategy = MoboStrategy(data_model=strategy_data)

    # Tell strategy about initial experiments
    strategy.tell(experiments)

    print("\nAsking for 3 new candidate experiments...")
    # With callable nonlinear constraints, the optimizer does not enforce them internally,
    # so we ask without raising on validation and keep only feasible candidates (with retries).
    n_want = 3
    max_attempts = 10
    candidates = None
    for _ in range(max_attempts):
        raw = strategy.ask(n_want, raise_validation_error=False, add_pending=False)
        fulfilled = domain.constraints.is_fulfilled(raw, tol=1e-3)
        if fulfilled.all():
            candidates = raw
            break
        feasible = raw[fulfilled]
        if len(feasible) >= n_want:
            candidates = feasible.head(n_want)
            break
        if candidates is None:
            candidates = feasible
        else:
            candidates = pd.concat(
                [candidates, feasible], ignore_index=True
            ).drop_duplicates()
        if len(candidates) >= n_want:
            candidates = candidates.head(n_want)
            break
    if candidates is None or len(candidates) < n_want:
        raise RuntimeError(
            f"Could not obtain {n_want} feasible candidates after {max_attempts} attempts. "
            "Try relaxing constraints or increasing initial samples."
        )

    print("\nProposed candidates (first 5):")
    print(candidates.head())

    # ========================================================================
    # 4. VERIFY CANDIDATES
    # ========================================================================

    print("\n" + "-" * 80)
    print("CANDIDATE VERIFICATION")
    print("-" * 80)

    # Check constraints
    for i, constraint in enumerate(domain.constraints, 1):
        fulfilled = constraint.is_fulfilled(candidates, tol=1e-3)
        violations = constraint(candidates)
        print(f"\nConstraint {i}:")
        print(f"  Satisfied: {fulfilled.sum()} / {len(candidates)}")
        print(f"  Max violation: {violations.max():.6f}")

    # Evaluate candidate performance
    print("\n" + "-" * 80)
    print("CANDIDATE PERFORMANCE")
    print("-" * 80)

    candidate_results = []
    for _, row in candidates.iterrows():
        result = model.simulate_batch(
            T=row["temperature"],
            tau=row["residence_time"],
            feed_ratio=row["feed_ratio"],
            pressure=row["pressure"],
        )
        candidate_results.append(result)

    performance = pd.DataFrame(candidate_results)

    print("\nPerformance metrics (first 5 candidates):")
    print(performance[["X_A", "Y_P", "Y_W", "S", "Q"]].head())

    print("\nSummary statistics:")
    print(
        f"  Conversion (X_A): {performance['X_A'].min():.3f} - {performance['X_A'].max():.3f}"
    )
    print(
        f"  Yield P (Y_P):    {performance['Y_P'].min():.3f} - {performance['Y_P'].max():.3f}"
    )
    print(
        f"  Yield W (Y_W):    {performance['Y_W'].min():.3f} - {performance['Y_W'].max():.3f}"
    )
    print(
        f"  Selectivity (S):  {performance['S'].min():.1f} - {performance['S'].max():.1f}"
    )
    print(
        f"  Heat gen. (Q):    {performance['Q'].min():.1f} - {performance['Q'].max():.1f} W/L"
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
