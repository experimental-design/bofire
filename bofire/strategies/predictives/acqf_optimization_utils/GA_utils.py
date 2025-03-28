from typing import List, Optional, Tuple, Type, Dict, Callable

import cvxopt
import numpy as np
import torch
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.repair import Repair as PymooRepair
from pymoo.core import variable as pymoo_variable
from torch import Tensor

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.domain.api import Domain
from bofire.data_models.types import InputTransformSpecs
from bofire.utils.torch_tools import (
    get_linear_constraints,
    get_nonlinear_constraints,
    tkwargs,
)

class OneHotEncodingToOrdinalEncoding:
    """Helper class for transfering bounds, and data from a domain with One-Hot-Encoded input
    features, to ordinal encoded features, as they are used for the GA """
    def __init__(self, domain: Domain, input_preprocessing_specs: InputTransformSpecs):

        self.bounds_org: List[Tuple[float, float]]
        self.bounds_transformed: List[Tuple[float, float]]
        self.transformer: List[Callable[[Tensor], Tensor]]


        for key in domain.inputs.get_keys():

            spec_ = input_preprocessing_specs.get(key)

            bounds_org = domain.inputs.get_by_key(key).get_bounds(spec_)
            self.bounds_org.append(bounds_org)

            replace = False
            if spec_ is not None:
                if spec_ == CategoricalEncodingEnum.ONE_HOT:
                    replace = True

            if not replace:
                self.bounds_transformed.append(bounds_org)

            else:
                bounds_transformed = domain.inputs.get_by_key(key).get_bounds(CategoricalEncodingEnum.ORDINAL)
                self.bounds_transformed.append(bounds_transformed)

                idx_transformed = sum([len(x[0]) for x in self.bounds_transformed]) - 1

                self.transformer.append(\
                    OneHotEncodingToOrdinalEncoding._transform_function_factory(idx_transformed, bounds_org))

    def get_bounds_transformed(self) -> Tuple[List[float], List[float]]:
        """return the new bounds, given ordinal-encodings"""
        lb, ub = [list(np.concatenate([x[i] for x in self.bounds_transformed])) for i in (0, 1)]
        return lb, ub

    def get_pymoo_types(self) -> List[pymoo_variable.Variable]:
        """return the pymoo types for the transformed input"""
        return [
            pymoo_variable.Real()
        ]

    def transform(self, x: Tensor) -> Tensor:
        """transform ordinal encoded input (from GA) to one-hot encoded input (for BoTorch model)

        Parameters:
            x: Tensor of shape (..., d_ord), where d is the number of input features in ordinal-encoded space

        Returns:
            x: Tensor, transformed input of shape (..., d)
        """
        for transform in self.transformer:
            x = transform(x)

        return x

    @staticmethod
    def _transform_function_factory(idx_transformed: int, bounds_org: Tuple[float, float]) \
            -> Callable[[Tensor], Tensor]:
        di = len(bounds_org[0])
        def _transform(x: Tensor) -> Tensor:
            """transform ordinal encoded input (from GA) to one-hot encoded input (for BoTorch model), for a single
            encoded column

            Parameters:
                x: Tensor of shape (..., d_ord), where d is the number of input features in ordinal-encoded space

            Returns:
                x: Tensor, transformed input of shape (..., d)
            """

            xl, xr = x[..., :idx_transformed], x[..., idx_transformed+1:]
            x_replace = x[..., idx_transformed]  # should be (0, 1, ...) ordinal encoded
            x_replace = torch.round(x_replace)
            x_new = torch.zeros(list(x.shape[:-1]) + [di])

            for i in range(di):
                mask_true = x_replace == i
                x_new[..., ~mask_true] = bounds_org[0][i]
                x_new[..., mask_true] = bounds_org[1][i]

            return torch.cat((xl, x_new, xr), dim=-1)

        return _transform


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
        constraints_include: Optional[List[Type[Constraint]]] = None,
    ):
        self.acqfs = acqfs

        self.encoding_switcher = OneHotEncodingToOrdinalEncoding(domain, input_preprocessing_specs)
        lb, ub = self.encoding_switcher.get_bounds_transformed()

        if (
            constraints_include is None
        ):  # we chould possibly extend this list in the future
            constraints_include = [
                ProductInequalityConstraint,
            ]
        else:
            assert all(c in (ProductInequalityConstraint,) for c in constraints_include)

        self.nonlinear_constraints = get_nonlinear_constraints(
            domain, includes=constraints_include
        )

        n_var = len(ub) * q
        xl = lb * q
        xu = ub * q
        self.d = len(lb)
        self.q = q

        # assert len(self.nonlinear_constraints) == 0, "To-Do: Nonlinear Constr."

        super().__init__(
            n_var=n_var,
            n_obj=len(acqfs),
            n_ieq_constr=len(self.nonlinear_constraints) * q,
            n_eq_constr=0,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False,
            vars=
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_pop = x.shape[0]
        x = torch.from_numpy(x).to(**tkwargs)
        x = x.reshape((n_pop, self.q, self.d))

        out["F"] = [-acqf(x).detach().numpy().reshape(-1) for acqf in self.acqfs]

        if self.nonlinear_constraints:
            G = []
            for constr in self.nonlinear_constraints:
                constr_val = -constr[0](x)  # converting to form g(x) <= 0
                G.append(constr_val.detach().numpy())

            out["G"] = np.hstack(G)


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

        self.constraints_include = constraints_include
        self.d = d
        self.q = q
        self.domain = domain
        self.bounds = bounds

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
        sol = cvxopt.solvers.qp(**self._create_qp_problem_input(X))
        x_corrected = np.array(sol["x"])
        X_corrected = x_corrected.reshape(X.shape)

        return X_corrected
