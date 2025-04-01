from typing import Dict, List, Optional, Tuple, Type

import cvxopt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from pymoo.core import variable as pymoo_variable
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableGA,
    MixedVariableMating,
)
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.repair import Repair as PymooRepair
from pymoo.termination import default as pymoo_default_termination
from torch import Tensor

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    ProductInequalityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.categorical import get_encoded_name
from bofire.data_models.strategies.api import (
    GeneticAlgorithm as GeneticAlgorithmDataModel,
)
from bofire.data_models.types import InputTransformSpecs
from bofire.utils.torch_tools import (
    get_linear_constraints,
    get_nonlinear_constraints,
    tkwargs,
)


class BofireDomainMixedVars:
    """Helper class for transfering bounds, and data from a domain with One-Hot-Encoded input
    features, to mixed-variable encoded features, as they are used for the GA
    see, e.g. https://pymoo.org/customization/mixed.html?highlight=variable%20type
    """

    def __init__(
        self, domain: Domain, input_preprocessing_specs: InputTransformSpecs, q: int
    ):
        self.domain = domain
        self.vars = {}
        self.q = q

        for key in domain.inputs.get_keys():
            spec_ = input_preprocessing_specs.get(key)

            bounds_org = np.array(
                domain.inputs.get_by_key(key).get_bounds(spec_)
            ).reshape(-1)

            replace = False
            if spec_ is not None:
                if spec_ == CategoricalEncodingEnum.ONE_HOT:
                    replace = True

            if not replace:
                self.vars[key] = pymoo_variable.Real(bounds=bounds_org)

            else:
                self.vars[key] = pymoo_variable.Choice(
                    options=domain.inputs.get_by_key(key).get_allowed_categories()
                )

    def pymoo_vars(self) -> Dict[str, pymoo_variable]:
        """return the variables in the format required by pymoo. Includes repeats for q-points"""
        vars = {}
        for qi in range(self.q):
            vars = {**vars, **{f"{key}_q{qi}": var for key, var in self.vars.items()}}
        return vars

    def _transform(self, X: List[dict]) -> np.ndarray:
        """Transform to numerical, encoded format: (n_pop, q, d), where:
        d is the (numerical, encoded) dimension
        """

        def _sub_array_key(key) -> np.ndarray:
            if isinstance(self.vars[key], pymoo_variable.Real):

                def _sub_array_key_qi(qi: int):
                    dict_key = f"{key}_q{qi}"
                    return np.array([xi[dict_key] for xi in X]).reshape((-1, 1, 1))

            else:

                def _sub_array_key_qi(qi: int) -> np.ndarray:
                    dict_key = f"{key}_q{qi}"
                    cat_vals = pd.Series([xi[dict_key] for xi in X])
                    enc_vals = (
                        self.domain.inputs.get_by_key(key)
                        .to_onehot_encoding(cat_vals)
                        .values
                    )
                    d_encoded = enc_vals.shape[1]
                    return enc_vals.reshape((-1, 1, d_encoded))

            return np.concatenate(
                [_sub_array_key_qi(qi) for qi in range(self.q)], axis=1
            )

        x = np.concatenate([_sub_array_key(key) for key in self.vars.keys()], axis=2)
        return x

    def transform_to_experiments(self, X: List[dict]) -> List[pd.DataFrame]:
        """Transform to a list of "experiments" dataframes for each q-point"""
        experiments = pd.DataFrame.from_records(X)
        q_column = np.array([int(x.split("_q")[-1]) for x in list(experiments)])

        experiments_out = []

        for qi in range(self.q):
            experiments_qi = experiments.iloc[:, q_column == qi]
            experiments_qi.columns = [x.split("_q")[0] for x in experiments_qi.columns]
            experiments_out.append(experiments_qi)

        return experiments_out

    def transform_mixed_to_botorch_domain(self, X: List[dict]) -> Tensor:
        """Transform the variables from the pymoo format to the format required by botorch, including one-hot encoding
        Will produce an n_pop x q x d tensor
        """
        return torch.from_numpy(self._transform(X)).to(**tkwargs)

    def transform_mixed_to_2D(self, X: List[dict]) -> np.ndarray:
        """Transform the variables from the mixed pymoo domain, to the form required by the 'repair' function, as
        2D matrix with shape (n_pop, d * q), where
            d is the (numerical, encoded) dimension

        """
        x = self._transform(X)
        return np.concatenate([x[:, i, :] for i in range(self.q)], axis=1)

    def _mixed_2D_idx(self) -> Dict[str, List[int]]:
        idx = {}
        last_idx: int = -1
        for qi in range(self.q):
            for key, pymoo_type_ in self.vars.items():
                if isinstance(pymoo_type_, pymoo_variable.Real):
                    idx[f"{key}_q{qi}"] = [last_idx + 1]
                    last_idx += 1
                elif isinstance(pymoo_type_, pymoo_variable.Choice):
                    idx[f"{key}_q{qi}"] = [
                        last_idx + 1 + i for i in range(len(pymoo_type_.options))
                    ]
                    last_idx += len(pymoo_type_.options)
        return idx

    def inverse_transform_to_mixed(self, X: np.ndarray) -> List[dict]:
        idx_map = self._mixed_2D_idx()
        out = pd.DataFrame(columns=list(idx_map))
        for key in list(self.vars):
            input_ref = self.domain.inputs.get_by_key(key)
            for qi in range(self.q):
                key_qi = f"{key}_q{qi}"
                idx = idx_map[key_qi]

                xi = X[:, np.array(idx)]

                if hasattr(input_ref, "from_onehot_encoding"):
                    cat_cols = [
                        get_encoded_name(input_ref.key, c) for c in input_ref.categories
                    ]
                    xi = input_ref.from_onehot_encoding(
                        pd.DataFrame(xi, columns=cat_cols)
                    ).values.reshape(-1)
                else:
                    xi = xi.reshape(-1)

                out[key_qi] = list(xi)

        # reformat to list of dicts
        return out.to_dict("records")


class AcqfOptimizationProblem(PymooProblem):
    """Transfers the acquisition function optimization problem on the bofire domain, into a pymoo-
    problem, which can be solved with e.g. the pymoo GA.

    The optimizer will handle one-hot encoded input features as
    """

    def __init__(
        self,
        acqfs,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        q: int,
        nonlinear_torch_constraints: Optional[List[Type[Constraint]]] = None,
        nonlinear_pandas_constraints: Optional[List[Constraint]] = None,
    ):
        self.acqfs = acqfs

        self.domain_handler = BofireDomainMixedVars(
            domain, input_preprocessing_specs, q
        )

        # torch constraints: evaluated in encoded space
        if nonlinear_torch_constraints is None:
            nonlinear_torch_constraints = [ProductInequalityConstraint]
        else:
            assert all(c in (ProductInequalityConstraint,) for c in nonlinear_torch_constraints)
        self.nonlinear_torch_constraints = get_nonlinear_constraints(
            domain, includes=nonlinear_torch_constraints
        )

        # pandas constraints: evaluated in original space
        if nonlinear_pandas_constraints is None:
            nonlinear_pandas_constraints = [NonlinearInequalityConstraint]
        else:
            assert all(c in (NonlinearInequalityConstraint,) for c in nonlinear_pandas_constraints)
        self.nonlinear_pandas_constraints = domain.constraints.get(
            includes=nonlinear_pandas_constraints)


        # assert len(self.nonlinear_constraints) == 0, "To-Do: Nonlinear Constr."

        super().__init__(
            n_obj=len(acqfs),
            n_ieq_constr=(len(self.nonlinear_torch_constraints) + len(self.nonlinear_pandas_constraints)) * q,
            n_eq_constr=0,
            elementwise_evaluation=False,
            vars=self.domain_handler.pymoo_vars(),
        )

    def _evaluate(self, x_ga_encoded, out, *args, **kwargs):
        x = self.domain_handler.transform_mixed_to_botorch_domain(x_ga_encoded)

        out["F"] = [-acqf(x).detach().numpy().reshape(-1) for acqf in self.acqfs]

        if self.nonlinear_torch_constraints:
            G = []
            for constr in self.nonlinear_torch_constraints:
                constr_val = -constr[0](x)  # converting to form g(x) <= 0
                G.append(constr_val.detach().numpy())

            out["G"] = np.hstack(G)

        if self.nonlinear_pandas_constraints:
            experiments = self.domain_handler.transform_to_experiments(x_ga_encoded)
            for constr in self.nonlinear_pandas_constraints:
                G = np.hstack([constr(exp).values.reshape((-1, 1)) for exp in experiments])
                out["G"] = np.hstack([out["G"], G]) if "G" in out else G


class LinearProjection(PymooRepair):
    """handles linear equality constraints by mapping to closest legal point in the design space
    using quadratic programming, by projecting to x':

        min(1/2 * ||(x'-x)||_2)

        s.t.
        A*x' = 0
        G*x' <= 0

    we will transform the Problem to the type:

        min( 1/2 * x'^T * P * x' + q*x')

        s.t.
            ...


    in order to solve this with the performant cvxopt.qp solver
    (https://cvxopt.org/userguide/coneprog.html#quadratic-programming)

    For performance, the problem is solved for the complete generation X = [x1 ; x2; ...]
    where x1, x2, ... are the vectors of each individual in the population and q-opt. points

    This class is passed to the pymoo algorithm class to correct the population after each generation

    """

    def __init__(
        self,
        domain: Domain,
        d: int,
        bounds: Tensor,
        q: int,
        domain_handler: Optional[BofireDomainMixedVars] = None,
        constraints_include: Optional[List[Type[Constraint]]] = None,
    ):
        if constraints_include is None:
            constraints_include = [LinearEqualityConstraint, LinearInequalityConstraint]
        else:
            for constr in constraints_include:
                assert constr in (
                    LinearEqualityConstraint,
                    LinearInequalityConstraint,
                ), "Only linear constraints supported for LinearProjection"

        self.domain_handler = domain_handler
        self.constraints_include = constraints_include
        self.d = d
        self.q = q
        self.domain = domain
        self.bounds = bounds

        cvxopt.solvers.options["show_progress"] = False

        super().__init__()

    def _create_qp_problem_input(self, X: np.ndarray) -> dict:
        n_pop = X.shape[0]
        n_x_points = n_pop * self.q

        def _eq_constr_to_list(eq_constr_) -> Tuple[List[int], List[float], float]:
            """decode "get_linear_constraints" output: x-index, coefficients, and b
            - convert from tensor to list of ints and floats
            - multiply with (-1) to adhere to (usual) cvxopt format A*x <= b, instead of Botorch A*x >= b
            """
            index: List[int] = [int(x) for x in (eq_constr_[0].detach().numpy())]
            coeffs: List[float] = list(-eq_constr_[1].detach().numpy())
            b: float = -eq_constr_[2]
            return index, coeffs, b

        eq_constr, ineq_constr = [], []
        if LinearEqualityConstraint in self.constraints_include:
            eq_constr = [
                _eq_constr_to_list(eq_constr_)
                for eq_constr_ in get_linear_constraints(
                    self.domain, LinearEqualityConstraint
                )
            ]
        if LinearInequalityConstraint in self.constraints_include:
            ineq_constr = [
                _eq_constr_to_list(eq_constr_)
                for eq_constr_ in get_linear_constraints(
                    self.domain, LinearInequalityConstraint
                )
            ]

        def repeated_blkdiag(m: cvxopt.matrix, N: int) -> cvxopt.spmatrix:
            """construct large matrix with block-diagolal matrix in the center of arbitrary size"""
            m_zeros = cvxopt.spmatrix([], [], [], m.size)
            return cvxopt.sparse(
                [[m_zeros] * i + [m] + [m_zeros] * (N - i - 1) for i in range(N)]
            )

        def _build_A_b_matrices_for_single_constr(
            index, coeffs, b
        ) -> Tuple[cvxopt.spmatrix, cvxopt.matrix]:
            """a single-line constraint matrix of the form A*x = b or A*x <= b"""
            A = cvxopt.spmatrix(coeffs, [0] * len(index), index, (1, self.d))
            b = cvxopt.matrix(b)
            return A, b

        def _build_A_b_matrices_for_n_points(
            constr: List[tuple],
        ) -> Tuple[cvxopt.spmatrix, cvxopt.matrix]:
            """build big sparse matrix for a constraint A*x = b, or A*x <= b (repeated A/b matrices for n_x_point"""

            if not constr:
                return cvxopt.spmatrix(
                    [], [], [], (0, self.d * n_x_points)
                ), cvxopt.matrix([], (0, 1), tc="d")

            # vertically combine all linear equality constr.
            Ab_single_eq = [
                _build_A_b_matrices_for_single_constr(*constr_) for constr_ in constr
            ]
            A = cvxopt.sparse([Ab[0] for Ab in Ab_single_eq])
            b = cvxopt.matrix([Ab[1] for Ab in Ab_single_eq])
            # repeat for each x in the population
            A = repeated_blkdiag(A, n_x_points)
            b = cvxopt.matrix([b] * n_x_points)
            return A, b

        def _build_G_h_for_box_bounds() -> Tuple[cvxopt.spmatrix, cvxopt.matrix]:
            """build linear inequality matrices, such that lb<=x<=ub -> G*x<=h"""
            G_bounds_ = cvxopt.sparse(
                [
                    cvxopt.spmatrix(1, range(self.d), range(self.d)),  # unity matrix
                    cvxopt.spmatrix(
                        -1, range(self.d), range(self.d)
                    ),  # negative unity matrix
                ]
            )
            lb, ub = (self.bounds[i, :].detach().numpy() for i in range(2))
            h_bounds_ = cvxopt.matrix(np.concatenate((ub.reshape(-1), -lb.reshape(-1))))
            G = repeated_blkdiag(G_bounds_, n_x_points)
            h = cvxopt.matrix([h_bounds_] * n_x_points)
            return G, h

        # Prepare Matrices for solving the estimation problem
        P = cvxopt.spmatrix(
            1.0, range(self.d * n_x_points), range(self.d * n_x_points)
        )  # the unit-matrix

        A, b = _build_A_b_matrices_for_n_points(eq_constr)
        G, h = _build_G_h_for_box_bounds()
        if ineq_constr:
            G_, h_ = _build_A_b_matrices_for_n_points(ineq_constr)
            G, h = cvxopt.sparse([G, G_]), cvxopt.matrix([h, h_])

        x = X.reshape(-1)
        q = cvxopt.matrix(-x)

        return {
            "P": P,
            "q": q,
            "G": G,
            "h": h,
            "A": A,
            "b": b,
            "initvals": cvxopt.matrix(x),
        }

    def _do(self, problem, X, **kwargs):
        if self.domain_handler is not None:
            X = self.domain_handler.transform_mixed_to_2D(X)

        sol = cvxopt.solvers.qp(**self._create_qp_problem_input(X))
        x_corrected = np.array(sol["x"])
        X_corrected = x_corrected.reshape(X.shape)

        if self.domain_handler is not None:
            X_corrected = self.domain_handler.inverse_transform_to_mixed(X_corrected)

        return X_corrected


def get_problem_and_algorithm(
    data_model: GeneticAlgorithmDataModel,
    domain: Domain,
    input_preprocessing_specs: InputTransformSpecs,
    acqfs: List[AcquisitionFunction],
    q: int,
    bounds_botorch_space: Tensor,
):
    # ===== Problem ====
    problem = AcqfOptimizationProblem(
        acqfs,
        domain,
        input_preprocessing_specs,
        q,
    )

    # ==== Algorithm ====
    algorithm_args = {
        "pop_size": data_model.population_size,
        # todo: other algorithm options, like n_offspring, crossover-functions etc.
    }

    # We handle linear equality constraint with a repair function
    repair_constraints = domain.constraints.get(
        includes=[LinearEqualityConstraint, LinearInequalityConstraint],
    )
    if len(repair_constraints) > 0:
        repair = LinearProjection(
            domain=domain,
            d=bounds_botorch_space.shape[1],
            bounds=bounds_botorch_space,
            q=q,
            domain_handler=problem.domain_handler,
        )

        algorithm_args["repair"] = repair
        algorithm_args["mating"] = MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination(), repair=repair
        )  # see https://github.com/anyoptimization/pymoo/issues/575

    algorithm = MixedVariableGA(**algorithm_args)

    termination_class = (
        pymoo_default_termination.DefaultSingleObjectiveTermination
        if len(acqfs) == 1
        else pymoo_default_termination.DefaultMultiObjectiveTermination
    )

    termination = termination_class(
        xtol=data_model.xtol,
        cvtol=data_model.cvtol,
        ftol=data_model.ftol,
        n_max_gen=data_model.n_max_gen,
        n_max_evals=data_model.n_max_evals,
    )

    return problem, algorithm, termination
