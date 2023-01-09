import math
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import validator
from pydantic.types import PositiveInt
from scipy.integrate import solve_ivp
from scipy.special import gamma

from bofire.benchmarks.benchmark import Benchmark
from bofire.domain import Domain
from bofire.domain.features import (
    ContinuousInput,
    ContinuousOutput,
    InputFeature,
    InputFeatures,
    OutputFeatures,
)
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective


class DTLZ2(Benchmark):
    """Multiobjective bechmark function for testing optimization algorithms.
    Info about the function: https://pymoo.org/problems/many/dtlz.html
    """

    def __init__(
        self, dim: PositiveInt, k: Optional[int], num_objectives: PositiveInt = 2
    ):
        self.num_objectives = num_objectives
        self.dim = dim
        self.k = k

        input_features = []
        for i in range(self.dim):
            input_features.append(
                ContinuousInput(key="x_%i" % (i), lower_bound=0.0, upper_bound=1.0)
            )
        output_features = []
        self.k = self.dim - self.num_objectives + 1
        for i in range(self.num_objectives):
            output_features.append(
                ContinuousOutput(key=f"f_{i}", objective=MinimizeObjective(w=1.0))
            )
        domain = Domain(
            input_features=InputFeatures(features=input_features),
            output_features=OutputFeatures(features=output_features),
        )
        self.ref_point = {
            feat: 1.1 for feat in domain.get_feature_keys(ContinuousOutput)
        }
        self._domain = domain

    @validator("dim")
    def validate_dim(cls, dim, values):
        num_objectives = values["num_objectives"]
        if dim <= values["num_objectives"]:
            raise ValueError(
                f"dim must be > num_objectives, but got {dim} and {num_objectives}."
            )
        return dim

    @property
    def best_possible_hypervolume(self) -> float:
        # hypercube - volume of hypersphere in R^d such that all coordinates are
        # positive
        hypercube_vol = self.ref_point[0] ** self.num_objectives  # type: ignore
        pos_hypersphere_vol = (
            math.pi ** (self.num_objectives / 2)
            / gamma(self.num_objectives / 2 + 1)
            / 2**self.num_objectives
        )
        return hypercube_vol - pos_hypersphere_vol

    def f(self, candidates):
        X = candidates[self.domain.get_feature_keys(InputFeature)].values  # type: ignore
        X_m = X[..., -self.k :]  # type: ignore
        g_X = ((X_m - 0.5) ** 2).sum(axis=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.num_objectives):
            idx = self.num_objectives - 1 - i
            f_i = g_X_plus1.copy()
            f_i *= np.cos(X[..., :idx] * pi_over_2).prod(axis=-1)
            if i > 0:
                f_i *= np.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        candidates[
            self.domain.output_features.get_keys_by_objective(excludes=None)  # type: ignore
        ] = np.stack(fs, axis=-1)
        candidates[
            [
                "valid_%s" % feat
                for feat in self.domain.output_features.get_keys_by_objective(  # type: ignore
                    excludes=None
                )
            ]
        ] = 1
        return candidates[self.domain.experiment_column_names].copy()  # type: ignore


class SnarBenchmark(Benchmark):
    """Nucleophilic substitution problem as a multiobjective test function for optimization algorithms."""

    def __init__(self, C_i: Optional[np.ndarray] = np.ndarray((1, 1))):
        """Initializes multiobjective test function object of type SnarBenchmark.

        Args:
            C_i (Optional[np.ndarray]): input concentrations
        """
        self.C_i = C_i

        # Decision variables
        # "residence time in minutes"
        input_features = [
            ContinuousInput(key="tau", lower_bound=0.5, upper_bound=2.0),
            # "equivalents of pyrrolidine"
            ContinuousInput(key="equiv_pldn", lower_bound=1.0, upper_bound=5.0),
            # "concentration of 2,4 dinitrofluorobenenze at reactor inlet (after mixing) in M"
            ContinuousInput(key="conc_dfnb", lower_bound=0.1, upper_bound=0.5),
            # "Reactor temperature in degress celsius"
            ContinuousInput(key="temperature", lower_bound=30, upper_bound=120.0),
        ]
        # Objectives
        # "space time yield (kg/m^3/h)"
        output_features = [
            ContinuousOutput(key="sty", objective=MaximizeObjective(w=1.0)),
            # "E-factor"
            ContinuousOutput(
                key="e_factor",
                objective=MinimizeObjective(w=1.0),
            ),
        ]
        self.ref_point = {"e_factor": 10.7, "sty": 2957.0}
        self._domain = Domain(
            input_features=InputFeatures(features=input_features),
            output_features=OutputFeatures(features=output_features),
        )

    @property
    def best_possible_hypervolume(self):
        return 10000.0

    def f(self, candidates):
        stys = []
        e_factors = []
        for i, candidate in candidates.iterrows():
            tau = float(candidate["tau"])
            equiv_pldn = float(candidate["equiv_pldn"])
            conc_dfnb = float(candidate["conc_dfnb"])
            T = float(candidate["temperature"])
            y, e_factor, res = self._integrate_equations(tau, equiv_pldn, conc_dfnb, T)
            stys.append(y)
            e_factors.append(e_factor)
            # candidates["sty"] = y
            # candidates["e_factor"] = e_factor

        candidates["sty"] = stys
        candidates["e_factor"] = e_factors
        candidates[
            [
                "valid_%s" % feat
                for feat in self.domain.output_features.get_keys_by_objective(  # type: ignore
                    excludes=None
                )
            ]
        ] = 1
        return candidates[self.domain.experiment_column_names].copy()  # type: ignore

    def _integrate_equations(self, tau, equiv_pldn, conc_dfnb, temperature, **kwargs):
        # Initial Concentrations in mM
        self.C_i = np.zeros(5)
        self.C_i[0] = conc_dfnb
        self.C_i[1] = equiv_pldn * conc_dfnb

        # Flowrate and residence time
        V = 5  # mL
        q_tot = V / tau
        # C1_0 = kwargs.get("C1_0", 2.0)  # reservoir concentration of 1 is 1 M = 1 mM
        # C2_0 = kwargs.get("C2_0", 4.2)  # reservoir concentration of  2 is 2 M = 2 mM
        # q_1 = self.C_i[0] / C1_0 * q_tot  # flowrate of 1 (dfnb)
        # q_2 = self.C_i[1] / C2_0 * q_tot  # flowrate of 2 (pldn)
        # q_eth = q_tot - q_1 - q_2  # flowrate of ethanol

        # Integrate
        res = solve_ivp(self._integrand, [0, tau], self.C_i, args=(temperature,))
        C_final = res.y[:, -1]

        # # Add measurement noise
        # C_final += (
        #     C_final * self.rng.normal(scale=self.noise_level, size=len(C_final)) / 100
        # )
        # C_final[
        #     C_final < 0
        # ] = 0  # prevent negative values of concentration introduced by noise

        # Calculate STY and E-factor
        M = [159.09, 71.12, 210.21, 210.21, 261.33]  # molecular weights (g/mol)
        sty = 6e4 / 1000 * M[2] * C_final[2] * q_tot / V  # convert to kg m^-3 h^-1
        if sty < 1e-6:
            sty = 1e-6
        rho_eth = 0.789  # g/mL (should adjust to temp, but just using @ 25C)
        term_2 = 1e-3 * sum([M[i] * C_final[i] * q_tot for i in range(5) if i != 2])
        if np.isclose(C_final[2], 0.0):
            # Set to a large value if no product formed
            e_factor = 1e3
        else:
            e_factor = (q_tot * rho_eth + term_2) / (1e-3 * M[2] * C_final[2] * q_tot)
        if e_factor > 1e3:
            e_factor = 1e3

        return sty, e_factor, {}

    def _integrand(self, t, C, T):
        # Kinetic Constants
        R = 8.314 / 1000  # kJ/K/mol
        T_ref = 90 + 273.71  # Convert to deg K
        T = T + 273.71  # Convert to deg K
        # Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1
        k = (
            lambda k_ref, E_a, temp: 0.6
            * k_ref
            * np.exp(-E_a / R * (1 / temp - 1 / T_ref))
        )
        k_a = k(57.9, 33.3, T)
        k_b = k(2.70, 35.3, T)
        k_c = k(0.865, 38.9, T)
        k_d = k(1.63, 44.8, T)

        # Reaction Rates
        r = np.zeros(5)
        for i in [0, 1]:  # Set to reactants when close
            C[i] = 0 if C[i] < 1e-6 * self.C_i[i] else C[i]  # type: ignore
        r[0] = -(k_a + k_b) * C[0] * C[1]
        r[1] = -(k_a + k_b) * C[0] * C[1] - k_c * C[1] * C[2] - k_d * C[1] * C[3]
        r[2] = k_a * C[0] * C[1] - k_c * C[1] * C[2]
        r[3] = k_a * C[0] * C[1] - k_d * C[1] * C[3]
        r[4] = k_c * C[1] * C[2] + k_d * C[1] * C[3]

        # Deltas
        dcdtau = r
        return dcdtau


class ZDT1(Benchmark):
    """ZDT1 function for testing optimization algorithms.
    Explanation of the function: https://datacrayon.com/posts/search-and-optimisation/practical-evolutionary-algorithms/synthetic-objective-functions-and-zdt1/
    """

    def __init__(self, n_inputs=30):
        """Initializes class of type ZDT1 which is a benchmark function for optimization problems.

        Args:
            n_inputs (int, optional): Number of inputs. Defaults to 30.
        """
        self.n_inputs = n_inputs
        input_features = [
            ContinuousInput(key=f"x{i+1}", lower_bound=0, upper_bound=1)
            for i in range(n_inputs)
        ]
        inputs = InputFeatures(features=input_features)
        output_features = [
            ContinuousOutput(key=f"y{i+1}", objective=MinimizeObjective(w=1))
            for i in range(2)
        ]
        outputs = OutputFeatures(features=output_features)
        self._domain = Domain(input_features=inputs, output_features=outputs)

    def f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self._domain.inputs.get_keys()[1:]].to_numpy()
        g = 1 + 9 / (self.n_inputs - 1) * np.sum(x, axis=1)
        y1 = X["x1"].to_numpy()
        y2 = g * (1 - (y1 / g) ** 0.5)
        return pd.DataFrame({"y1": y1, "y2": y2}, index=X.index)

    def get_optima(self, points=100):
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.sqrt(x)], axis=1)
        return pd.DataFrame(y, columns=self.domain.outputs.get_keys())
