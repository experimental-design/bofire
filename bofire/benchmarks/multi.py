import json
import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.test_functions.multi_objective import ZDT1 as BotorchZDT1
from pydantic import field_validator
from pydantic.types import PositiveInt
from scipy.integrate import solve_ivp
from scipy.special import gamma

import bofire.surrogates.api as surrogates
from bofire.benchmarks.benchmark import Benchmark
from bofire.benchmarks.data.aniline_cn_crosscoupling import (
    EXPERIMENTS as ANNILINE_CN_CROSSCOUPLING_EXPERIMENTS,
)
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
    Input,
)
from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
)
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.utils.torch_tools import tkwargs


class DTLZ2(Benchmark):
    """Multiobjective benchmark function for testing optimization algorithms.
    Info about the function: https://pymoo.org/problems/many/dtlz.html
    """

    def __init__(self, dim: PositiveInt, num_objectives: PositiveInt = 2, **kwargs):
        """Initializes object of Type DTLZ2 which is a benchmark function.

        Args:
            dim (PositiveInt): Dimension of input vector
            num_objectives (PositiveInt, optional): Dimension of output vector. Defaults to 2.
            **kwargs: Additional arguments for the Benchmark class.

        """
        super().__init__(**kwargs)
        self.num_objectives = num_objectives
        self.dim = dim

        inputs = []
        for i in range(self.dim):
            inputs.append(ContinuousInput(key="x_%i" % (i), bounds=[0, 1]))
        outputs = []
        self.k = self.dim - self.num_objectives + 1
        for i in range(self.num_objectives):
            outputs.append(
                ContinuousOutput(key=f"f_{i}", objective=MinimizeObjective(w=1.0)),
            )
        domain = Domain(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=outputs),
        )
        self.ref_point = {
            feat: 1.1 for feat in domain.outputs.get_keys(ContinuousOutput)
        }
        self._domain = domain

    @field_validator("dim")
    @classmethod
    def validate_dim(cls, dim, values):
        num_objectives = values["num_objectives"]
        if dim <= values["num_objectives"]:
            raise ValueError(
                f"dim must be > num_objectives, but got {dim} and {num_objectives}.",
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

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Function evaluation of DTLZ2.

        Args:
            candidates (pd.DataFrame): Input vector for x-values. Columns go from x0 to xdim.

        Returns:
            pd.DataFrame: Function values in output vector. Columns are f0 and f1.

        """
        X = candidates[self.domain.inputs.get_keys(Input)].values
        X_m = X[..., -self.k :]
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

        col_names = self.domain.outputs.get_keys_by_objective(
            includes=MinimizeObjective,
        )
        y_values = np.stack(fs, axis=-1)
        Y = pd.DataFrame(data=y_values, columns=col_names)
        Y[
            [
                "valid_%s" % feat
                for feat in self.domain.outputs.get_keys_by_objective(
                    includes=MinimizeObjective,
                )
            ]
        ] = 1
        return Y


class BNH(Benchmark):
    def __init__(self, constraints: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.constraints = constraints

        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x1", bounds=[0, 5]),
                    ContinuousInput(key="x2", bounds=[0, 3]),
                ],
            ),
            outputs=Outputs(
                features=[
                    ContinuousOutput(key="f1", objective=MinimizeObjective(w=1.0)),
                    ContinuousOutput(key="f2", objective=MinimizeObjective(w=1.0)),
                ],
            ),
        )
        if self.constraints:
            self._domain.outputs.features.append(  # type: ignore
                ContinuousOutput(
                    key="c1",
                    objective=MinimizeSigmoidObjective(tp=25, steepness=1000),
                ),
            )
            self._domain.outputs.features.append(  # type: ignore
                ContinuousOutput(
                    key="c2",
                    objective=MaximizeSigmoidObjective(tp=7.7, steepness=1000),
                ),
            )

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        experiments = candidates.eval("f1=4*x1**2 + 4*x2**2", inplace=False)
        experiments = experiments.eval("f2=(x1-5)**2 + (x2-5)**2", inplace=False)
        experiments["valid_f1"] = 1
        experiments["valid_f2"] = 1
        if not self.constraints:
            return experiments[["f1", "f2", "valid_f1", "valid_f2"]].copy()
        experiments = experiments.eval("c1=(x1-5)**2 + x2**2", inplace=False)
        experiments = experiments.eval("c2=(x1-8)**2 + (x2+3)**2", inplace=False)
        experiments["valid_c1"] = 1
        experiments["valid_c2"] = 1
        return experiments[
            ["f1", "f2", "c1", "c2", "valid_c1", "valid_c2", "valid_f1", "valid_f2"]
        ].copy()


class TNK(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x1", bounds=[0, math.pi]),
                    ContinuousInput(key="x2", bounds=[0, math.pi]),
                ],
            ),
            outputs=Outputs(
                features=[
                    ContinuousOutput(key="f1", objective=MinimizeObjective(w=1.0)),
                    ContinuousOutput(key="f2", objective=MinimizeObjective(w=1.0)),
                    ContinuousOutput(
                        key="c1",
                        objective=MaximizeSigmoidObjective(tp=0.0, steepness=500),
                    ),
                    ContinuousOutput(
                        key="c2",
                        objective=MinimizeSigmoidObjective(tp=0.5, steepness=500),
                    ),
                ],
            ),
        )

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        experiments = candidates.eval("f1=x1", inplace=False)
        experiments = experiments.eval("f2=x2", inplace=False)
        experiments = experiments.eval(
            "c1=x1**2 + x2**2 -1 -0.1*cos(16*arctan(x1/x2))",
            inplace=False,
        )
        experiments = experiments.eval("c2=(x1-0.5)**2+(x2-0.5)**2", inplace=False)
        experiments["valid_c1"] = 1
        experiments["valid_c2"] = 1
        experiments["valid_f1"] = 1
        experiments["valid_f2"] = 1
        return experiments[
            ["f1", "f2", "c1", "c2", "valid_c1", "valid_c2", "valid_f1", "valid_f2"]
        ].copy()


class C2DTLZ2(DTLZ2):
    """Constrained DTLZ2 benchmark function. Taken from
    https://github.com/pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py.
    """

    def __init__(self, dim: PositiveInt, num_objectives: PositiveInt = 2, **kwargs):
        super().__init__(dim, num_objectives, **kwargs)
        # add also the constraint
        self._domain.outputs.features.append(  # type: ignore
            ContinuousOutput(
                key="slack",
                objective=MaximizeSigmoidObjective(w=1.0, tp=0, steepness=1.0 / 1e-3),
            ),
        )

    @property
    def best_possible_hypervolume(self) -> float:
        return 0.3996406303723544

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        r = 0.2
        Y = super()._f(candidates=candidates)
        # here the constrained is calculated
        # TODO port it to numpy
        f_X = torch.from_numpy(
            Y[self.domain.outputs.get_keys_by_objective(MinimizeObjective)].values,
        )
        term1 = (f_X - 1).pow(2)
        mask = ~(torch.eye(f_X.shape[-1], device=f_X.device).bool())
        indices = torch.arange(f_X.shape[1], device=f_X.device).repeat(f_X.shape[1], 1)
        indexer = indices[mask].view(f_X.shape[1], f_X.shape[-1] - 1)
        term2_inner = (
            f_X.unsqueeze(1)
            .expand(f_X.shape[0], f_X.shape[-1], f_X.shape[-1])
            .gather(dim=-1, index=indexer.repeat(f_X.shape[0], 1, 1))
        )
        term2 = (term2_inner.pow(2) - r**2).sum(dim=-1)
        min1 = (term1 + term2).min(dim=-1).values
        min2 = ((f_X - 1 / math.sqrt(f_X.shape[-1])).pow(2) - r**2).sum(dim=-1)
        slack = pd.Series(
            -torch.min(min1, min2).unsqueeze(-1).squeeze().numpy(),
            name="slack",
        )
        Y = pd.concat([Y, slack], axis=1)
        Y["valid_slack"] = 1
        return Y


class SnarBenchmark(Benchmark):
    """Nucleophilic aromatic substitution problem as a multiobjective test function for optimization algorithms.
    Solving of a differential equation system with varying initial values.
    """

    def __init__(self, C_i: Optional[np.ndarray] = None, **kwargs):
        """Initializes multiobjective test function object of type SnarBenchmark.

        Args:
            C_i (Optional[np.ndarray]): Input concentrations. Defaults to [1, 1]
            **kwargs: Additional arguments for the Benchmark class.

        """
        super().__init__(**kwargs)
        if C_i is None:
            C_i = np.array([1, 1])
        self.C_i = C_i

        # Decision variables
        # "residence time in minutes"
        inputs = [
            ContinuousInput(key="tau", bounds=[0.5, 2]),
            # "equivalents of pyrrolidine"
            ContinuousInput(key="equiv_pldn", bounds=[1, 5]),
            # "concentration of 2,4 dinitrofluorobenenze at reactor inlet (after mixing) in M"
            ContinuousInput(key="conc_dfnb", bounds=[0.1, 0.5]),
            # "Reactor temperature in degrees celsius"
            ContinuousInput(key="temperature", bounds=[30, 120]),
        ]
        # Objectives
        # "space time yield (kg/m^3/h)"
        outputs = [
            ContinuousOutput(key="sty", objective=MaximizeObjective(w=1.0)),
            # "E-factor"
            ContinuousOutput(
                key="e_factor",
                objective=MinimizeObjective(w=1.0),
            ),
        ]
        self.ref_point = {"e_factor": 10.7, "sty": 2957.0}
        self._domain = Domain(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=outputs),
        )

    @property
    def best_possible_hypervolume(self):
        return 10000.0

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Function evaluation. Returns output vector.

        Args:
            candidates (pd.DataFrame): Input vector. Columns: tau, equiv_pldn, conc_dfnb, temperature

        Returns:
            pd.DataFrame: Output vector. Columns: sty, e_factor

        """
        stys = []
        e_factors = []
        for _, candidate in candidates.iterrows():
            tau = float(candidate["tau"])
            equiv_pldn = float(candidate["equiv_pldn"])
            conc_dfnb = float(candidate["conc_dfnb"])
            T = float(candidate["temperature"])
            y, e_factor, res = self._integrate_equations(tau, equiv_pldn, conc_dfnb, T)
            stys.append(y)
            e_factors.append(e_factor)
            # candidates["sty"] = y
            # candidates["e_factor"] = e_factor

        # return only y values instead of appending them to input dataframe
        Y = pd.DataFrame({"sty": stys, "e_factor": e_factors})
        Y[
            [
                "valid_%s" % feat
                for feat in self.domain.outputs.get_keys_by_objective(excludes=None)
            ]
        ] = 1
        return Y

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
        sty = max(sty, 1e-6)
        rho_eth = 0.789  # g/mL (should adjust to temp, but just using @ 25C)
        term_2 = 1e-3 * sum([M[i] * C_final[i] * q_tot for i in range(5) if i != 2])
        if np.isclose(C_final[2], 0.0):
            # Set to a large value if no product formed
            e_factor = 1e3
        else:
            e_factor = (q_tot * rho_eth + term_2) / (1e-3 * M[2] * C_final[2] * q_tot)
        e_factor = min(e_factor, 1e3)

        return sty, e_factor, {}

    def _integrand(self, t, C, T):
        # Kinetic Constants
        R = 8.314 / 1000  # kJ/K/mol
        T_ref = 90 + 273.71  # Convert to deg K
        T = T + 273.71  # Convert to deg K
        # Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1

        def k(k_ref, E_a, temp):
            return 0.6 * k_ref * np.exp(-E_a / R * (1 / temp - 1 / T_ref))

        k_a = k(57.9, 33.3, T)
        k_b = k(2.70, 35.3, T)
        k_c = k(0.865, 38.9, T)
        k_d = k(1.63, 44.8, T)

        # Reaction Rates
        r = np.zeros(5)
        for i in [0, 1]:  # Set to reactants when close
            C[i] = 0 if C[i] < 1e-6 * self.C_i[i] else C[i]
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

    def __init__(self, n_inputs=30, **kwargs):
        """Initializes class of type ZDT1 which is a benchmark function for optimization problems.

        Args:
            n_inputs (int, optional): Number of inputs. Defaults to 30.
            **kwargs: Additional arguments for the Benchmark class.

        """
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        inputs = [
            ContinuousInput(key=f"x{i+1}", bounds=[0, 1]) for i in range(n_inputs)
        ]
        inputs = Inputs(features=inputs)
        outputs = [
            ContinuousOutput(key=f"y{i+1}", objective=MinimizeObjective(w=1))
            for i in range(2)
        ]
        outputs = Outputs(features=outputs)
        self._domain = Domain(inputs=inputs, outputs=outputs)
        self.zdt = BotorchZDT1(dim=n_inputs)

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        """Function evaluation.

        Args:
            X (pd.DataFrame): Input values

        Returns:
            pd.DataFrame: Function values. Columns are y1, y2, valid_y1 and valid_y2.

        """
        Xt = torch.from_numpy(X.values).to(**tkwargs)
        Y = self.zdt(Xt).numpy()
        return pd.DataFrame(
            {"y1": Y[:, 0], "y2": Y[:, 1], "valid_y1": 1, "valid_y2": 1},
            index=X.index,
        )

    def get_optima(self, points=100) -> pd.DataFrame:
        """Pareto front of the output variables.

        Args:
            points (int, optional): Number of points of the pareto front. Defaults to 100.

        Returns:
            pd.DataFrame: 2D pareto front with x and y values.

        """
        x = np.linspace(0, 1, points)
        y = np.stack([x, 1 - np.sqrt(x)], axis=1)
        return pd.DataFrame(y, columns=self.domain.outputs.get_keys())


class CrossCoupling(Benchmark):
    """Cross Coupling adapted from Summit (https://github.com/sustainable-processes/summit)

    Virtual experiments representing the Aniline Cross-Coupling reaction
    similar to Baumgartner et al. (2019) <https://`doi.org/10.1021/acs.oprd.9b00236>.
    Experimental outcomes are based on a SingleTaskGP fitted on the experimental data published by Baumgartner et al.
    This is a five dimensional optimisation of temperature, residence time, base equivalents,
    catalyst and base.
    The categorical variables (catalyst and base) contain descriptors
    calculated using COSMO-RS. Specifically, the descriptors are the first two sigma moments.

    Args:
        Benchmark (Benchmark): Benchmark base class

    """

    def __init__(
        self,
        **kwargs,
    ):
        # "residence time in minutes"
        inputs = [
            CategoricalDescriptorInput(
                key="catalyst",
                categories=["tBuXPhos", "tBuBrettPhos", "AlPhos"],
                descriptors=[
                    "area_cat",
                    "M2_cat",
                ],  # , 'M3_cat', 'Macc3_cat', 'Mdon3_cat'] #,'mol_weight', 'sol']
                values=[
                    [
                        460.7543,
                        67.2057,
                    ],  # 30.8413, 2.3043, 0], #, 424.64, 421.25040226],
                    [
                        518.8408,
                        89.8738,
                    ],  # 39.4424, 2.5548, 0], #, 487.7, 781.11247064],
                    [
                        819.933,
                        129.0808,
                    ],  # 83.2017, 4.2959, 0], #, 815.06, 880.74916884],
                ],
            ),
            CategoricalDescriptorInput(
                key="base",
                categories=["TEA", "TMG", "BTMG", "DBU"],
                descriptors=[
                    "area",
                    "M2",
                ],  # , 'M3', 'Macc3', 'Mdon3', 'mol_weight', 'sol'
                values=[
                    [162.2992, 25.8165],  # 40.9469, 3.0278, 0], #101.19, 642.2973283],
                    [
                        165.5447,
                        81.4847,
                    ],  # 107.0287, 10.215, 0.0169], # 115.18, 534.01544123],
                    [
                        227.3523,
                        30.554,
                    ],  # 14.3676, 1.1196, 0.0127], # 171.28, 839.81215],
                    [192.4693, 59.8367],  # 82.0661, 7.42, 0], # 152.24, 1055.82799],
                ],
            ),
            # "base equivalents"
            ContinuousInput(key="base_eq", bounds=[1, 2.5]),
            # "Reactor temperature in degrees celsius"
            ContinuousInput(key="temperature", bounds=[30, 100]),
            # "residence time in seconds (s)"
            ContinuousInput(key="t_res", bounds=[60, 1800]),
        ]

        input_preprocessing_specs = {
            "catalyst": CategoricalEncodingEnum.DESCRIPTOR,
            "base": CategoricalEncodingEnum.DESCRIPTOR,
        }

        # Objectives: yield and cost
        outputs = [
            ContinuousOutput(
                key="yield",
                objective=MaximizeObjective(w=1.0, bounds=[0.0, 1.0]),
            ),
            ContinuousOutput(
                key="cost",
                objective=MinimizeObjective(w=1.0, bounds=[0.0, 1.0]),
            ),
        ]
        self.ref_point = {"yield": 0.0, "cost": 1.0}

        self._domain = Domain(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=outputs),
        )

        data = pd.DataFrame.from_dict(json.loads(ANNILINE_CN_CROSSCOUPLING_EXPERIMENTS))

        data = data.rename(columns={"base_equivalents": "base_eq", "yld": "yield"})
        data["valid_yield"] = 1

        data_model = SingleTaskGPSurrogate(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=[outputs[0]]),
            input_preprocessing_specs=input_preprocessing_specs,  # type: ignore
        )
        ground_truth_yield = surrogates.map(data_model)

        ground_truth_yield.fit(experiments=data)  # type: ignore
        self.ground_truth_yield = ground_truth_yield
        super().__init__(**kwargs)

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Function evaluation. Returns output vector.

        Args:
            candidates (pd.DataFrame): Input vector. Columns: catalyst, base, base_eq, temperature, t_res

        Returns:
            pd.DataFrame: Output vector. Columns: yield, cost, valid_yield, valid_cost

        """
        costs = self._calculate_costs(candidates)
        yields = self.ground_truth_yield.predict(candidates)

        Y = pd.concat([yields["yield_pred"], pd.Series(costs, name="cost")], axis=1)
        Y.rename(columns={"yield_pred": "yield"}, inplace=True)

        Y[
            [
                "valid_%s" % feat
                for feat in self.domain.outputs.get_keys_by_objective(excludes=None)
            ]
        ] = 1
        return Y

    def _calculate_costs(self, conditions):
        """Function to calculate the overall costs of a recipe

        Args:
            conditions (pd.DataFrame): The suggested candidate experiments

        Returns:
            np.array: Vector with costs of suggested candidates

        """
        catalyst = conditions["catalyst"].values
        base = conditions["base"].values
        base_equiv = conditions["base_eq"].values

        # Calculate amounts
        droplet_vol = 40 * 1e-3  # mL
        mmol_triflate = 0.91 * droplet_vol
        mmol_anniline = 1.6 * mmol_triflate
        catalyst_equiv = {
            "tBuXPhos": 0.0095,
            "tBuBrettPhos": 0.0094,
            "AlPhos": 0.0094,
        }
        mmol_catalyst = [catalyst_equiv[c] * mmol_triflate for c in catalyst]
        mmol_base = base_equiv * mmol_triflate

        # Calculate costs
        cost_triflate = mmol_triflate * 5.91  # triflate is $5.91/mmol
        cost_anniline = mmol_anniline * 0.01  # anniline is $0.01/mmol
        cost_catalyst = np.array(
            [self._get_catalyst_cost(c, m) for c, m in zip(catalyst, mmol_catalyst)],
        )
        cost_base = np.array(
            [self._get_base_cost(b, m) for b, m in zip(base, mmol_base)],
        )
        tot_cost = cost_triflate + cost_anniline + cost_catalyst + cost_base
        if len(tot_cost) == 1:
            tot_cost = tot_cost[0]
        return tot_cost

    def _get_catalyst_cost(self, catalyst, catalyst_mmol):
        """Function to calculate the catalyst costs

        Args:
            catalyst (str): Catalyst name
            catalyst_mmol (float): Amount of catalyst used

        Returns:
            float: Catalyst costs

        """
        catalyst_prices = {
            "tBuXPhos": 94.08,
            "tBuBrettPhos": 182.85,
            "AlPhos": 594.18,
        }
        return float(catalyst_prices[catalyst] * catalyst_mmol)

    def _get_base_cost(self, base, mmol_base):
        """Function to calculate the base costs

        Args:
            base (str): Base name
            mmol_base (float): Amount of base used

        Returns:
            float: Base costs

        """
        # prices in $/mmol
        base_prices = {
            "DBU": 0.03,
            "BTMG": 1.2,
            "TMG": 0.001,
            "TEA": 0.01,
        }
        return float(base_prices[base] * mmol_base)
