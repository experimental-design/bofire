from abc import abstractmethod
from typing import List, Optional, Tuple, Type

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import sparse

from bofire.data_models.constraints import api as constraints
from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import CategoricalInput, ContinuousInput
from bofire.data_models.types import InputTransformSpecs
from bofire.utils.torch_tools import get_linear_constraints


def default_input_preprocessing_specs(
    domain: Domain,
) -> InputTransformSpecs:
    """Default input preprocessing specs for the GA optimizer: If none given, will use Ordinal encoding for all categorical inputs"""
    return {
        key: CategoricalEncodingEnum.ORDINAL
        for key in domain.inputs.get_keys(CategoricalInput)
    }


class DomainRepair:
    """Abstract class for all methods, considering 'repair' of experiments"""

    @abstractmethod
    def __call__(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """
        Converting a set of experiments in a domain, into the feasible space
        """

    @abstractmethod
    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        """Checks if a constraint is implemented.

        Args:
            my_type (Type[Feature]): The type of the constraint.

        Returns:
            bool: True if the constraint is implemented, False otherwise.

        """
        pass

    @abstractmethod
    def validate_domain(self, domain: Domain):
        """Validates the fit of the domain to the optimizer.

        Args:
            domain (Domain): The domain to be validated.
        """
        pass


class LinearProjection(DomainRepair):
    """handles linear equality constraints by mapping to closest feasible point in the design space
    using quadratic programming, by projecting to x':

        min(1/2 * ||(x'-x)||_2)

        s.t.
        A*x' = 0
        G*x' <= 0
        lb <= x' <= ub

    we will transform the Problem to the type:

        min( 1/2 * x'^T * P * x' + q*x')

        s.t.
            ...

    The scaled problem will instead solve for z = (x - lb) / (ub - lb), where z in [0, 1]

    in order to solve this with the performant cvxpy solver. (https://www.cvxpy.org/examples/basic/quadratic_program.html)

    For performance, the problem is solved for the complete generation X = [x1 ; x2; ...]
    where x1, x2, ... are the vectors of each individual in the population and q-opt. points

    NChooseK-constraints can be added to the problem:
    - the lower bound of the largest n_non_zero elements in each experiment is set to min_delta
    - the upper bound of the smallest n_zero elements in each experiment is set to zero

    This class is passed to the pymoo algorithm class to correct the population after each generation

    Args:
        domain: The domain containing input features and constraints to be repaired.
        q: Number of candidates to generate in batch mode. Defaults to 1.
        input_preprocessing_specs: Specifications for preprocessing input features. If None,
            default preprocessing specs are generated from the domain.
        constraints_include: List of constraint types to include in the repair process.
            Supported types are LinearEqualityConstraint, LinearInequalityConstraint,
            and NChooseKConstraint. If None, all three constraint types are included.
        n_choose_k_constr_min_delta: Minimum delta value for enforcing NChooseK constraints.
            The lower bound of the largest n_non_zero elements is set to this value. Defaults to 1e-3.
        verbose: If True, enables verbose logging of the repair process. Defaults to False.

    """

    def is_constraint_implemented(self, my_type: Type[constraints.Constraint]) -> bool:
        return my_type in (
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NChooseKConstraint,
        )

    def __init__(
        self,
        domain: Domain,
        q: int = 1,
        input_preprocessing_specs: Optional[InputTransformSpecs] = None,
        constraints_include: Optional[List[Type[Constraint]]] = None,
        n_choose_k_constr_min_delta: float = 1e-3,
        verbose: bool = False,
        scale_problem: bool = True,
    ):
        if constraints_include is None:
            constraints_include = [
                LinearEqualityConstraint,
                LinearInequalityConstraint,
                NChooseKConstraint,
            ]
        else:
            for constr in constraints_include:
                assert constr in (
                    LinearEqualityConstraint,
                    LinearInequalityConstraint,
                    NChooseKConstraint,
                ), "Only linear constraints and NChooseK supported for LinearProjection"

        # get bounds
        if input_preprocessing_specs is None:
            input_preprocessing_specs = default_input_preprocessing_specs(domain)
        self.input_preprocessing_specs = input_preprocessing_specs

        lower, upper = domain.inputs.get_bounds(
            specs=input_preprocessing_specs,
        )
        bounds = np.vstack(
            (np.array(lower).reshape((1, -1)), np.array(upper).reshape((1, -1)))
        )
        self.bounds = bounds

        def lin_constr_to_list(constr_) -> Tuple[List[int], List[float], float]:
            """decode "get_linear_constraints" output: x-index, coefficients, and b
            - convert from tensor to list of ints and floats
            - multiply with (-1) to adhere to (usual) cvxopt format A*x <= b, instead of Botorch A*x >= b
            """
            index: List[int] = [int(x) for x in (constr_[0].detach().numpy())]
            coeffs: List[float] = list(-constr_[1].detach().numpy())
            b: float = -constr_[2]
            return index, coeffs, b

        # linear constraints
        self.eq_constr, self.ineq_constr = [], []
        if LinearEqualityConstraint in constraints_include:
            self.eq_constr = [
                lin_constr_to_list(eq_constr_)
                for eq_constr_ in get_linear_constraints(
                    domain, LinearEqualityConstraint
                )
            ]
        if LinearInequalityConstraint in constraints_include:
            self.ineq_constr = [
                lin_constr_to_list(ineq_constr_)
                for ineq_constr_ in get_linear_constraints(
                    domain, LinearInequalityConstraint
                )
            ]

        # NChooseKConstrains
        class NChooseKBoundProjection:
            """helper class for correcting upper and lower bounds to fulfill NChooseK constraints
            in QP projection"""

            def __init__(
                self,
                constraints: List[NChooseKConstraint],
                bounds: np.ndarray,
                min_delta,
            ):
                self.lb, self.ub = (
                    bounds[0, :].reshape((1, -1)),
                    bounds[1, :].reshape((1, -1)),
                )
                self.min_delta = min_delta
                self.d = bounds.shape[1]

                self.n_zero, self.n_non_zero, self.idx = [], [], []
                for constraint in constraints:
                    di = len(constraint.features)
                    self.idx.append(
                        np.array(
                            [
                                domain.inputs.get_keys(ContinuousInput).index(key)
                                for key in constraint.features
                            ]
                        )
                    )
                    self.n_zero.append(di - constraint.max_count)
                    self.n_non_zero.append(constraint.min_count)

            @staticmethod
            def _ub_correction(
                ub: np.ndarray, x: np.ndarray, n_zero: int
            ) -> np.ndarray:
                """correct upper bounds: set the upper bound of the smallest n_zero elements in each row to zero"""
                if n_zero == 0:
                    return ub

                # Get indices of the lowest n_zero values per row
                low_indices = np.argsort(x, axis=1)[:, :n_zero]

                # set the lowest indices of each row to zero
                rows = np.arange(x.shape[0])[:, None]
                ub[rows, low_indices] = 0

                return ub

            @staticmethod
            def _lb_correction(
                lb: np.ndarray, x: np.ndarray, n_non_zero: int, min_delta: float
            ) -> np.ndarray:
                """correct lower bounds: set the lower bound of the largest n_non_zero elements in each row to min_delta"""
                if n_non_zero == 0:
                    return lb

                # Get indices of largest n_non_zero values for each row
                top_indices = np.argsort(x, axis=1)[:, -n_non_zero:]

                # Set all values in lb to 0 initially
                lb.fill(0)
                rows = np.arange(x.shape[0])[:, None]
                lb[rows, top_indices] = min_delta

                return lb

            def __call__(self, x: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
                """will generate bounds for the QP projection for all n_pop*q vectors of x, with
                dimensions (n_pop, d*q)

                Returns:
                    List[Tuple[np.ndarray, np.ndarray]]: lower and upper bounds for each x in the population

                """
                x = x.reshape((-1, self.d))

                lb, ub = self.lb.copy(), self.ub.copy()
                lb, ub = (
                    np.repeat(lb, x.shape[0], axis=0),
                    np.repeat(ub, x.shape[0], axis=0),
                )

                for n_zero, n_non_zero, idx in zip(
                    self.n_zero, self.n_non_zero, self.idx
                ):
                    x_, lb_, ub_ = x[:, idx], lb[:, idx].copy(), ub[:, idx].copy()
                    lb_ = self._lb_correction(lb_, x_, n_non_zero, self.min_delta)
                    ub_ = self._ub_correction(ub_, x_, n_zero)

                    lb[:, idx], ub[:, idx] = lb_, ub_

                return [(lb[i, :], ub[i, :]) for i in range(x.shape[0])]

        self.n_choose_k_constr = None
        if NChooseKConstraint in constraints_include:
            n_choose_k_constraints = domain.constraints.get(
                includes=[NChooseKConstraint]
            )
            if n_choose_k_constraints.constraints:
                self.n_choose_k_constr = NChooseKBoundProjection(
                    n_choose_k_constraints.constraints,  # type: ignore
                    bounds,
                    n_choose_k_constr_min_delta,
                )

        # self.domain_handler = domain_handler
        self.d = bounds.shape[1]
        self.q = q
        self.domain = domain
        self.bounds = bounds
        self.verbose = verbose
        self.scale_problem = scale_problem

    def _create_qp_problem_input(self, X: np.ndarray) -> dict:
        n_pop = X.shape[0]
        n_x_points = n_pop * self.q

        def _build_A_b_matrices_for_single_constr(
            index, coeffs, b
        ) -> Tuple[sparse.csr_array, np.ndarray]:
            """a single-line constraint matrix of the form A*x = b or A*x <= b"""
            A = sparse.csr_array(
                (
                    [float(ci) for ci in coeffs],
                    (
                        [0] * len(index),  # row index, only one row
                        [int(idx) for idx in index],  # column indices
                    ),
                ),
                shape=(1, self.d),
            )
            b = np.array(b).reshape((1, 1))
            return A, b

        def _build_A_b_matrices_for_n_points(
            constr: List[Tuple[List[int], List[float], float]],
        ) -> Tuple[sparse.csr_array, np.ndarray]:
            """build big sparse matrix for a constraint A*x = b, or A*x <= b (repeated A/b matrices for n_x_point)

            will build a large constraint matrix A, with block-diagonal structure, such that each block represents
            the constraint A*x =/<= b for one x in the population and q-points.

            Args:
                constr (List[Tuple[List[int], List[float], float]]): list of constraints, each as tuple of
                    (index, coefficients, b) as returned by get_linear_constraints



            """

            if not constr:
                return sparse.csr_array((0, self.d * n_x_points)), np.zeros(
                    shape=(0, 1)
                )

            # vertically combine all linear equality constr.
            Ab_single_eq = [
                _build_A_b_matrices_for_single_constr(*constr_) for constr_ in constr
            ]
            A = sparse.vstack([Ab[0] for Ab in Ab_single_eq])
            b = np.vstack([Ab[1] for Ab in Ab_single_eq])
            # repeat for each x in the population
            A = sparse.block_diag([A] * n_x_points)
            b = np.vstack([b] * n_x_points)
            return A, b

        def _build_G_h_for_box_bounds() -> Tuple[sparse.csr_array, np.ndarray]:
            """build linear inequality matrices, such that lb<=x<=ub -> G*x<=h:

            G = [I; -I]
            h = [ub; -lb]

            """
            G_bounds_ = sparse.vstack(
                [
                    sparse.identity(self.d),  # unity matrix
                    -sparse.identity(self.d),  # negative unity matrix
                ]
            )
            G = sparse.block_diag([G_bounds_] * n_x_points)

            if self.n_choose_k_constr is None:  # use the normal lb/ub
                lb, ub = (self.bounds[i, :] for i in range(2))
                h_bounds_ = np.concatenate((ub.reshape(-1), -lb.reshape(-1)))
                h = np.vstack([h_bounds_.reshape(-1, 1)] * n_x_points)
            else:
                # correct bounds for NChooseK constraints
                bounds = self.n_choose_k_constr(X)
                # alternate upper- and (-1*lower) bound
                h = np.vstack(
                    [np.concatenate((b[1], -b[0])).reshape((-1, 1)) for b in bounds]
                )

            return G, h

        # Prepare Matrices for solving the estimation problem
        P = sparse.identity(self.d * n_x_points)  # the unit-matrix: same for scaled and unscaled problem

        A, b = _build_A_b_matrices_for_n_points(self.eq_constr)
        G, h = _build_G_h_for_box_bounds()
        if self.ineq_constr:
            G_, h_ = _build_A_b_matrices_for_n_points(self.ineq_constr)
            G = sparse.vstack([G, G_])
            h = np.vstack([h, h_])

        x = X.reshape(-1)

        if self.scale_problem:
            scale = np.clip(self.bounds[1, :] - self.bounds[0, :], a_min=1e-3, a_max=np.inf)
            scale = np.repeat(scale.reshape((1, -1)), n_x_points, axis=0).reshape(-1)
            intercept = np.repeat(self.bounds[0, :].reshape((1, -1)), n_x_points, axis=0).reshape(-1)
            self.scale, self.intercept = scale, intercept

            b = b - A @ intercept.reshape((-1, 1))
            A = A @ sparse.diags(scale, 0)

            h = h - G @ intercept.reshape((-1, 1))
            G = G @ sparse.diags(scale, 0)
            
            x = (x - intercept)/scale

        q = -x

        return {
            "P": P,
            "q": q,
            "G": G,
            "h": h,
            "A": A,
            "b": b,
            "initvals": x,
        }

    def solve_numeric(self, X: np.ndarray) -> np.ndarray:
        """
        Converting a set of experiments in a domain, into the feasible space, considering the inputs in the numerical
        space. inputs must be in the right orde (according to the order in domain.inputs.get_keys())
        """

        matrices = self._create_qp_problem_input(X)

        x_var = cp.Variable((np.size(X), 1))
        x_var.value = matrices["initvals"].reshape(-1, 1)

        objective = cp.Minimize(
            0.5 * cp.quad_form(x_var, matrices["P"]) + matrices["q"].T @ x_var
        )
        constraints = []
        if matrices["A"].shape[0] > 0:
            constraints.append(matrices["A"] @ x_var == matrices["b"])
        if matrices["G"].shape[0] > 0:
            constraints.append(matrices["G"] @ x_var <= matrices["h"])

        problem = cp.Problem(objective, constraints)

        problem.solve(verbose=self.verbose)

        x_corrected = x_var.value
        if self.scale_problem:
            x_corrected = self.intercept.reshape(x_corrected.shape) + x_corrected * self.scale.reshape(x_corrected.shape)

        X_corrected = x_corrected.reshape(X.shape)
        return X_corrected


    def __call__(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """
        Converting a set of experiments in a domain, into the feasible space
        """
        X = self.domain.inputs.transform(
            experiments, self.input_preprocessing_specs
        ).values

        X_corrected = self.solve_numeric(X)

        return self.domain.inputs.inverse_transform(
            experiments=pd.DataFrame(
                X_corrected, columns=self.domain.inputs.get_keys()
            ),
            specs=self.input_preprocessing_specs,
        )
