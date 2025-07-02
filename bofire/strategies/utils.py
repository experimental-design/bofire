from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core import variable as pymoo_variable
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableGA,
    MixedVariableMating,
)
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.repair import Repair as PymooRepair
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import default as pymoo_default_termination
from scipy import sparse
from torch import Tensor

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearInequalityConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.strategies.api import (
    GeneticAlgorithmOptimizer as GeneticAlgorithmDataModel,
)
from bofire.data_models.types import InputTransformSpecs
from bofire.utils.torch_tools import (
    get_linear_constraints,
    get_nonlinear_constraints,
    tkwargs,
)


def _default_input_preprocessing_specs(
    domain: Domain,
) -> InputTransformSpecs:
    """Default input preprocessing specs for the GA optimizer: If none given, will use OneHot encoding for all categorical inputs"""
    input_preprocessing_specs = {}
    for input_ in domain.inputs.get():
        if isinstance(input_, CategoricalDescriptorInput):
            input_preprocessing_specs[input_.key] = CategoricalEncodingEnum.DESCRIPTOR
        elif isinstance(input_, CategoricalInput):
            input_preprocessing_specs[input_.key] = CategoricalEncodingEnum.ONE_HOT
    return input_preprocessing_specs


class GaMixedDomainHandler:
    """Helper class for transferring bounds, and data from different (encoded) input
    features, to mixed-variable encoded features, as they are used for the GA
    see, e.g. https://pymoo.org/customization/mixed.html?highlight=variable%20type.

    Transformations in this class are:
        (1) from the "original" non-encoded domain of the problem to the mixed-GA domain
        (2) from the encoded "botorch" domain (e.g. including one-hot, and descriptor encodings) to the mixed-GA domain

    The GA optimizes all q-experiments in a batch simultaneuously. The variables for the pymoo optimizer are:

        [feature1_q0, feature2_q0, ..., feature1_q1, feature2_q1, ....].

    Features, which are encoded in the input_preprocessing_specs are transformed to different pymoo types:
        - Categoric variables (descriptor- or one-hot-encoded) -> 'Choice'
        - Numerical discrete -> 'Integer'

    For the evaluation of the objective function and constraints, the variables in the GA-domain are transferred back
    to the other domains.

    Args:
        domain (Domain): problem domain
            input_preprocessing_specs (InputTransformSpecs): transformation specs, as they are needed for transforming the inputs
            from the domain to the numeric (torch) domain
        q (int): Number of experiments.

    """

    def __init__(
        self,
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        q: int = 1,
    ):
        self.domain = domain
        self.vars = {}
        self.pymoo_conversion: Dict[
            str, dict
        ] = {}  # conversions, which are not handled by the "domain"
        self.input_preprocessing_specs = input_preprocessing_specs
        self.q = q

        if input_preprocessing_specs is None:
            input_preprocessing_specs = _default_input_preprocessing_specs(domain)

        for key in domain.inputs.get_keys():
            spec_ = input_preprocessing_specs.get(key)
            input_ref = domain.inputs.get_by_key(key)

            if isinstance(input_ref, ContinuousInput) and spec_ is None:
                # simple case: non-transformed continuous variable
                bounds_org = np.array(
                    domain.inputs.get_by_key(key).get_bounds(spec_)  # type: ignore
                ).reshape(-1)
                self.vars[key] = pymoo_variable.Real(
                    bounds=bounds_org
                )  # default variable

            elif (
                isinstance(input_ref, CategoricalDescriptorInput)
                and spec_ == CategoricalEncodingEnum.DESCRIPTOR
            ):
                # categorical descriptor
                self.vars[key] = pymoo_variable.Choice(
                    options=input_ref.get_allowed_categories(),
                )

            elif (
                isinstance(input_ref, CategoricalInput)
                and spec_ == CategoricalEncodingEnum.ONE_HOT
            ):
                # one-hot encoded categorical input
                self.vars[key] = pymoo_variable.Choice(
                    options=input_ref.get_allowed_categories(),
                )

            elif isinstance(input_ref, DiscreteInput) and spec_ is None:
                # numerical discrete input
                conversion = dict(enumerate(input_ref.values))
                self.vars[key] = pymoo_variable.Integer(bounds=[0, len(conversion) - 1])
                self.pymoo_conversion[key] = conversion

            else:
                raise NotImplementedError(
                    f"Input {input_ref.key}: Input type {type(input_ref)} with preprocessing spec {spec_} not supported"
                )

    @property
    def pymoo_conversion_inverse(self) -> Dict[str, dict]:
        return {
            factor: {value: key for (key, value) in conv_dict.items()}
            for (factor, conv_dict) in self.pymoo_conversion.items()
        }

    @property
    def column_name_mapping(self) -> Dict[str, List[str]]:
        """Mapping of original column name to ..._qi columns with q-points repeats"""
        return {
            key: [f"{key}_q{qi}" for qi in range(self.q)]
            for key in self.domain.inputs.get_keys()
        }

    @property
    def column_name_mapping_inverse(self) -> Dict[str, str]:
        """Mapping of ..._qi columns to original column name"""
        return {
            f"{key}_q{qi}": key
            for key in self.domain.inputs.get_keys()
            for qi in range(self.q)
        }

    @property
    def column_name_mapping_inverse_qindex(self) -> Dict[str, int]:
        """Mapping of ..._qi columns to index of q_point"""
        return {
            f"{key}_q{qi}": qi
            for key in self.domain.inputs.get_keys()
            for qi in range(self.q)
        }

    def pymoo_vars(self) -> Dict[str, pymoo_variable.Variable]:
        """return the variables in the format required by pymoo. Includes repeats for q-points.

        Returns:
            Dict[str, pymoo_variable.Variable]: dictionary, to be used in pymoo.Problem __init__. The keys

        """
        vars = {}
        for qi in range(self.q):
            vars = {
                **vars,
                **{
                    self.column_name_mapping[key][qi]: var
                    for key, var in self.vars.items()
                },
            }
        return vars

    def _pymoo_specific_transform(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """handles the non-domain encodings (e.g. pymoo-type 'Integer' to discrete"""
        for key, conversion in self.pymoo_conversion.items():
            experiments[key] = experiments[key].replace(conversion)
        return experiments

    def _transform(self, X: List[dict]) -> np.ndarray:
        """Transform from pymoo mixed List[dict]-format to numerical, encoded format: (n_pop, q, d), where
        d is the (numerical, encoded) dimension
        """

        experiments = self.transform_to_experiments_per_q_point(X)
        x_numeric = [
            self.domain.inputs.transform(
                ex_, self.input_preprocessing_specs
            ).values.astype(float)
            for ex_ in experiments
        ]
        x_numeric = np.concatenate([np.expand_dims(x, 1) for x in x_numeric], axis=1)
        return x_numeric

    def transform_to_experiments_per_q_point(self, X: List[dict]) -> List[pd.DataFrame]:
        """Transform to a list of "experiments" into dataframes for each q-point. Each of the q dataframes contains
        as many rows as individuals in the population, and as many columns as input features.
        """
        experiments = pd.DataFrame.from_records(X)
        q_column = np.array(
            [self.column_name_mapping_inverse_qindex[x] for x in list(experiments)]
        )

        experiments_out = []

        for qi in range(self.q):
            experiments_qi = experiments.iloc[:, q_column == qi]
            experiments_qi.columns = [
                self.column_name_mapping_inverse[x] for x in experiments_qi.columns
            ]
            experiments_qi = self._pymoo_specific_transform(experiments_qi)
            experiments_out.append(experiments_qi)

        return experiments_out

    def transform_to_experiments_per_individual(
        self, X: List[dict]
    ) -> List[pd.DataFrame]:
        """Transform to a list of dataframes. Will return a dataframe for each individual. Each of the n_pop dataframes
        contains as many rows as q-points, and as many columns as input features."""
        experiments = pd.DataFrame.from_records(X)
        q_column = np.array(
            [self.column_name_mapping_inverse_qindex[x] for x in list(experiments)]
        )
        columns = [self.column_name_mapping_inverse[key] for key in list(experiments)]

        def format_experiment(row: pd.Series) -> pd.DataFrame:
            """Format a single row of the experiment to a DataFrame with q-points as rows"""
            # Create a DataFrame with the values and the corresponding q-point
            exp = pd.DataFrame({"value": row, "q": q_column, "column": columns})
            return exp.pivot(index="q", columns="column", values="value").reset_index(
                drop=True
            )

        return [format_experiment(row) for _, row in experiments.iterrows()]

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
                    idx[self.column_name_mapping[key][qi]] = [last_idx + 1]
                    last_idx += 1
                elif isinstance(pymoo_type_, pymoo_variable.Choice):
                    idx[self.column_name_mapping[key][qi]] = [
                        last_idx + 1 + i for i in range(len(pymoo_type_.options))
                    ]
                    last_idx += len(pymoo_type_.options)
        return idx

    def inverse_transform_to_mixed(self, X: np.ndarray) -> List[dict]:
        """Transform from numeric 2D format to pymoo mixed format"""

        d = int(X.shape[1] / self.q)
        q_ranges = [range(i * d, (i + 1) * d) for i in range(self.q)]
        x_numeric = [X[:, idx] for idx in q_ranges]

        columns = list(
            self.domain.inputs.transform(
                pd.DataFrame(columns=list(self.vars)),
                self.input_preprocessing_specs,
            )
        )

        experiments = [
            self.domain.inputs.inverse_transform(
                pd.DataFrame(x, columns=columns), self.input_preprocessing_specs
            )
            for x in x_numeric
        ]

        # pymoo-type conversion
        for i, experiment in enumerate(experiments):
            for key, conversion_dict in self.pymoo_conversion_inverse.items():
                experiments[i][key] = [conversion_dict[x] for x in experiment[key]]

        # appending "_qi" to headers
        for i, experiment in enumerate(experiments):
            experiment.columns = [
                self.column_name_mapping[col][i] for col in list(experiment)
            ]

        # concatenate
        experiments = pd.concat(experiments, axis=1)

        # reformat to list of dicts
        return experiments.to_dict("records")


class DomainOptimizationProblem(PymooProblem):
    """Transfers the optimization problem on the bofire domain, into a pymoo-
    problem, which can be solved with e.g. the pymoo GA.

    The optimizer will handle encoded input features as different pymoo-types (Choice), as the model. The problem
    contains the function for evaluating the objective functions, including constraints.
        - Transformation from the mixed-variable type domain, into the numeric torch domain
        - Evaluation of objective functions, such as acquisition functions
        - Evaluation of constraints:
            Nonlinear inequality constraints are evaluated in the mixed-domain (using pandas 'eval')
            Other constraints are evaluated in the numeric torch-domain

    Some constraints are not evaluated in the objective function: They are handled in the 'repair' function. See
    "LinearProjection" class for details.

    Args:
        objective_callables: List of objective functions, which are evaluated in the
            optimization problem. It should return a tensor or numpy-array of shape (n,).
            Alternatively, the functions can return a matrix of shape (n, ny_i), where ny_i is the number of outputs. In
            this case, the total number of outputs (sum(ny_i)) must be specified in the `n_obj` parameter.

            The callable_format defines, if the callables are evaluated in the numeric torch domain, or in the "original"
            pandas domain.
            If "torch", the callables should accept a torch tensor of shape (n, q, d), where n is the
            number of individuals, q is the number of experiments in each batch, and d is the number of the (numeric
            encoded) input features.
            If "pandas", the callables should accept a list of pandas DataFrames, where each DataFrame contains "q"
            rows and "d" columns, where "d" is the number of (notencoded) input features.
            Each DataFrame corresponds to one individual of the GA.

        domain (Domain): The bofire domain, which contains the input and output features.
        input_preprocessing_specs (InputTransformSpecs): The input preprocessing specifications, which are used to
            transform the inputs to the correct format for the models. This is needed for the variable transformation
            from the mixed-variable type domain to the numeric torch domain.
        q (int): Number of experiments in each batch.
        callable_format (Literal["torch", "pandas"]): Format of the objective callables. see `objective_callables`
            for details.
        nonlinear_torch_constraints (Optional[List[Type[Constraint]]]): List of types of nonlinear torch constraints, which are
            evaluated in the numeric torch domain. Must be transformable by the "get_nonlinear_constraints" function.
            Cannot be used, if `callable_format` is "pandas".
        nonlinear_pandas_constraints (Optional[List[Type[Constraint]]]): List of types nonlinear pandas constraints, which are
            evaluated in the original domain.
        n_obj (Optional[int]): Number of objectives. If not specified, it will be inferred from the length of `objective_callables`.
            Assuming that each callable returns a single output. If the callables return multiple outputs, n_obj
            must be set to the sum of the number of outputs of each callable.

    """

    def __init__(
        self,
        objective_callables: List[
            Union[
                Callable[[Tensor], Tensor], Callable[[List[pd.DataFrame]], np.ndarray]
            ]
        ],
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        q: int = 1,
        callable_format: Literal["torch", "pandas"] = "torch",
        nonlinear_torch_constraints: Optional[List[Type[Constraint]]] = None,
        nonlinear_pandas_constraints: Optional[List[Type[Constraint]]] = None,
        n_obj: Optional[int] = None,
    ):
        self.objective_callables = objective_callables
        self.domain_handler = GaMixedDomainHandler(domain, input_preprocessing_specs, q)
        self.callable_format = callable_format

        # torch constraints: evaluated in encoded space
        if nonlinear_torch_constraints is None:
            nonlinear_torch_constraints = [ProductInequalityConstraint]
        else:
            assert all(
                c in (ProductInequalityConstraint,) for c in nonlinear_torch_constraints
            )
        self.nonlinear_torch_constraints = get_nonlinear_constraints(
            domain, includes=nonlinear_torch_constraints
        )
        if self.nonlinear_torch_constraints and callable_format == "pandas":
            raise ValueError(
                "Nonlinear torch constraints cannot be used with callable_format='pandas'."
            )

        # pandas constraints: evaluated in original space
        if nonlinear_pandas_constraints is None:
            nonlinear_pandas_constraints = [NonlinearInequalityConstraint]
        else:
            assert all(
                c in (NonlinearInequalityConstraint,)
                for c in nonlinear_pandas_constraints
            )
        self.nonlinear_pandas_constraints = domain.constraints.get(
            includes=nonlinear_pandas_constraints
        )

        super().__init__(
            n_obj=n_obj or len(self.objective_callables),
            n_ieq_constr=(
                len(self.nonlinear_torch_constraints)
                + len(self.nonlinear_pandas_constraints)
            )
            * q,
            n_eq_constr=0,
            elementwise_evaluation=False,
            vars=self.domain_handler.pymoo_vars(),
        )

    def _evaluate(self, x_ga_encoded: List[dict], out: dict, *args, **kwargs):
        obj = []
        if self.callable_format == "torch":
            x = self.domain_handler.transform_mixed_to_botorch_domain(x_ga_encoded)
        elif self.callable_format == "pandas":
            x = self.domain_handler.transform_to_experiments_per_individual(
                x_ga_encoded
            )
        else:
            raise NotImplementedError(f"unknown callable_format {self.callable_format}")

        # evaluate objectives
        for ofnc in self.objective_callables:
            ofnc_val = np.array([])
            if self.callable_format == "torch":
                ofnc_val = ofnc(x).detach().numpy()  # type: ignore
            elif self.callable_format == "pandas":
                ofnc_val = ofnc(x)  # type: ignore

            if ofnc_val.ndim == 1:  # type: ignore
                obj.append(ofnc_val.reshape(-1))
            elif ofnc_val.ndim == 2:
                obj += [ofnc_val[:, i] for i in range(ofnc_val.shape[1])]
            else:
                raise ValueError(
                    f"Objective function {ofnc} returned an invalid shape: {ofnc_val.shape}"
                )

        out["F"] = obj

        if self.nonlinear_torch_constraints:
            G = []
            for constr in self.nonlinear_torch_constraints:
                # converting to form g(x) <= 0
                constr_val = -constr[0](x)  # type: ignore
                G.append(constr_val.detach().numpy())  # type: ignore

            out["G"] = np.hstack(G)

        if self.nonlinear_pandas_constraints:
            experiments = self.domain_handler.transform_to_experiments_per_q_point(
                x_ga_encoded
            )
            for constr in self.nonlinear_pandas_constraints:
                G = np.hstack(
                    [constr(exp).values.reshape((-1, 1)) for exp in experiments]
                )
                out["G"] = np.hstack([out["G"], G]) if (out["G"] is not None) else G


class LinearProjection(PymooRepair):
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


    in order to solve this with the performant cvxpy solver. (https://www.cvxpy.org/examples/basic/quadratic_program.html)

    For performance, the problem is solved for the complete generation X = [x1 ; x2; ...]
    where x1, x2, ... are the vectors of each individual in the population and q-opt. points

    NChooseK-constraints can be added to the problem:
    - the lower bound of the largest n_non_zero elements in each experiment is set to min_delta
    - the upper bound of the smallest n_zero elements in each experiment is set to zero

    This class is passed to the pymoo algorithm class to correct the population after each generation

    """

    def __init__(
        self,
        domain: Domain,
        d: int,
        bounds: Tensor,
        q: int,
        domain_handler: Optional[GaMixedDomainHandler] = None,
        constraints_include: Optional[List[Type[Constraint]]] = None,
        n_choose_k_constr_min_delta: float = 1e-3,
        verbose: bool = False,
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

                # sort each row, and set the smallest n_zero elements to zero
                ub[np.argsort(x) < n_zero] = 0
                return ub

            @staticmethod
            def _lb_correction(
                lb: np.ndarray, x: np.ndarray, n_non_zero: int, min_delta: float
            ) -> np.ndarray:
                """correct upper bounds: set the upper bound of the smallest n_zero elements in each row to zero"""
                if n_non_zero == 0:
                    return lb

                # sort each row, and set the largest n_non_zero elements to min_delta
                d = x.shape[1]
                lb[np.argsort(x) >= d - n_non_zero] = min_delta
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
                    bounds.detach().numpy(),
                    n_choose_k_constr_min_delta,
                )

        self.domain_handler = domain_handler
        self.d = d
        self.q = q
        self.domain = domain
        self.bounds = bounds
        self.verbose = verbose

        super().__init__()

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
                lb, ub = (self.bounds[i, :].detach().numpy() for i in range(2))
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
        P = sparse.identity(self.d * n_x_points)  # the unit-matrix

        A, b = _build_A_b_matrices_for_n_points(self.eq_constr)
        G, h = _build_G_h_for_box_bounds()
        if self.ineq_constr:
            G_, h_ = _build_A_b_matrices_for_n_points(self.ineq_constr)
            G = sparse.vstack([G, G_])
            h = np.vstack([h, h_])

        x = X.reshape(-1)
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

    def _do(self, problem, X, **kwargs):
        if self.domain_handler is not None:
            X = self.domain_handler.transform_mixed_to_2D(X)

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
        X_corrected = x_corrected.reshape(X.shape)

        if self.domain_handler is not None:
            X_corrected = self.domain_handler.inverse_transform_to_mixed(X_corrected)

        return X_corrected


def get_torch_bounds_from_domain(
    domain: Domain, input_preprocessing_specs: InputTransformSpecs
) -> torch.Tensor:
    """Get the bounds for the optimization problem in the format required by BoTorch."""
    lower, upper = domain.inputs.get_bounds(
        specs=input_preprocessing_specs,
    )
    return torch.tensor([lower, upper]).to(**tkwargs)


def get_ga_problem_and_algorithm(
    data_model: GeneticAlgorithmDataModel,
    domain: Domain,
    objective_callables: List[
        Union[Callable[[Tensor], Tensor], Callable[[List[pd.DataFrame]], np.ndarray]]
    ],
    q: int = 1,
    callable_format: Literal["torch", "pandas"] = "torch",
    input_preprocessing_specs: Optional[InputTransformSpecs] = None,
    n_obj: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[
    DomainOptimizationProblem,
    MixedVariableGA,
    Union[
        pymoo_default_termination.DefaultMultiObjectiveTermination,
        pymoo_default_termination.DefaultSingleObjectiveTermination,
    ],
]:
    """Convenience function to generate all pymoo- classes, needed for the optimization of the acquisition function(s)

    Args:
        data_model (GeneticAlgorithmDataModel): specifications for the algorithm
        domain (Domain): optimization domain
        objective_callables (List[Union[Callable[[Tensor], Tensor], Callable[[List[pd.DataFrame]], np.ndarray]]]):
            List of objective functions, which are evaluated in the
            optimization problem. It should return a tensor or numpy-array of shape (n,).
            Alternatively, the functions can return a matrix of shape (n, ny_i), where ny_i is the number of outputs. In
            this case, the total number of outputs (sum(ny_i)) must be specified in the `n_obj` parameter.

            The callable_format defines, if the callables are evaluated in the numeric torch domain, or in the "original"
            pandas domain.
            If "torch", the callables should accept a torch tensor of shape (n, q, d), where n is the
            number of individuals, q is the number of experiments in each batch, and d is the number of the (numeric
            encoded) input features.
            If "pandas", the callables should accept a list of pandas DataFrames, where each DataFrame contains "q"
            rows and "d" columns, where "d" is the number of (notencoded) input features.
            Each DataFrame corresponds to one individual of the GA.
        q (int): number of experiments
        callable_format (Literal["torch", "pandas"]): Format of the objective callables. see `objective_callables`
        input_preprocessing_specs (InputTransformSpecs): specification of the encoding types, used in the acqfs
                objective_callables (List[Callable[[Tensor], Tensor]]): List of objective functions, which are evaluated in the
            optimization problem. The functions should be callable with a tensor of shape (n, q, d), where d is the
            number of features, and q is the number of experiments. It should return a tensor of shape (n,).
            Alternatively, the functions can return a matrix of shape (n, ny_i), where ny_i is the number of outputs. In
            this case, the total number of outputs (sum(ny_i)) must be specified in the `n_obj` parameter.

        n_obj (Optional[int]): Number of objectives. If not specified, it will be inferred from the length of `objective_callables`.
            Assuming that each callable returns a single output. If the callables return multiple outputs, n_obj
            must be set to the sum of the number of outputs of each callable.
        verbose (bool, optional): Whether to print the QP iterations. Defaults to False.

    Returns
        problem
        algorithm
        termination
    """

    if input_preprocessing_specs is None:
        input_preprocessing_specs = _default_input_preprocessing_specs(domain)

    bounds_botorch_space = get_torch_bounds_from_domain(
        domain, input_preprocessing_specs
    )

    n_obj = n_obj or len(objective_callables)

    # ===== Problem ====
    problem = DomainOptimizationProblem(
        objective_callables,
        domain,
        input_preprocessing_specs,
        q,
        callable_format=callable_format,
        n_obj=n_obj,
    )

    # ==== Algorithm ====
    algorithm_args = {
        "pop_size": data_model.population_size,
    }

    # We handle linear equality constraint with a repair function
    repair_constraints = domain.constraints.get(
        includes=[
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NChooseKConstraint,
        ],
    )
    if len(repair_constraints) > 0:
        repair = LinearProjection(
            domain=domain,
            d=bounds_botorch_space.shape[1],
            bounds=bounds_botorch_space,
            q=q,
            domain_handler=problem.domain_handler,
            verbose=verbose,
        )

        algorithm_args["repair"] = repair
        algorithm_args["mating"] = MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination(), repair=repair
        )  # see https://github.com/anyoptimization/pymoo/issues/575

    if n_obj > 1:
        algorithm_args["survival"] = RankAndCrowdingSurvival()

    algorithm = MixedVariableGA(**algorithm_args)

    termination_class = (
        pymoo_default_termination.DefaultSingleObjectiveTermination
        if n_obj == 1
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


def run_ga(
    data_model: GeneticAlgorithmDataModel,
    domain: Domain,
    objective_callables: List[
        Union[Callable[[Tensor], Tensor], Callable[[List[pd.DataFrame]], np.ndarray]]
    ],
    q: int = 1,
    callable_format: Literal["torch", "pandas"] = "torch",
    input_preprocessing_specs: Optional[InputTransformSpecs] = None,
    n_obj: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Union[Tensor, List[pd.DataFrame]], Union[Tensor, np.ndarray]]:
    """Convenience function to minimize one or multiple objective functions using a genetic algorithm.

    Args:
        data_model (GeneticAlgorithmDataModel): specifications for the algorithm
        domain (Domain): optimization domain
        objective_callables (List[Union[Callable[[Tensor], Tensor], Callable[[List[pd.DataFrame]], np.ndarray]]]):
            List of objective functions, which are evaluated in the
            optimization problem. It should return a tensor or numpy-array of shape (n,).
            Alternatively, the functions can return a matrix of shape (n, ny_i), where ny_i is the number of outputs. In
            this case, the total number of outputs (sum(ny_i)) must be specified in the `n_obj` parameter.

            The callable_format defines, if the callables are evaluated in the numeric torch domain, or in the "original"
            pandas domain.
            If "torch", the callables should accept a torch tensor of shape (n, q, d), where n is the
            number of individuals, q is the number of experiments in each batch, and d is the number of the (numeric
            encoded) input features.
            If "pandas", the callables should accept a list of pandas DataFrames, where each DataFrame contains "q"
            rows and "d" columns, where "d" is the number of (notencoded) input features.
            Each DataFrame corresponds to one individual of the GA.
        q (int): number of experiments
        callable_format (Literal["torch", "pandas"]): Format of the objective callables. see `objective_callables`
        input_preprocessing_specs (InputTransformSpecs): specification of the encoding types, used in the acqfs
                objective_callables (List[Callable[[Tensor], Tensor]]): List of objective functions, which are evaluated in the
            optimization problem. The functions should be callable with a tensor of shape (n, q, d), where d is the
            number of features, and q is the number of experiments. It should return a tensor of shape (n,).
            Alternatively, the functions can return a matrix of shape (n, ny_i), where ny_i is the number of outputs. In
            this case, the total number of outputs (sum(ny_i)) must be specified in the `n_obj` parameter.

        n_obj (Optional[int]): Number of objectives. If not specified, it will be inferred from the length of `objective_callables`.
            Assuming that each callable returns a single output. If the callables return multiple outputs, n_obj
            must be set to the sum of the number of outputs of each callable.
        verbose (bool, optional): Whether to print the QP iterations. Defaults to False.

    Returns
        x_opt (Union[Tensor, pd.DataFrame]): optimized experiments
        f_opt (np.ndarray): objective function value
    """

    problem, algorithm, termination = get_ga_problem_and_algorithm(
        data_model=data_model,
        domain=domain,
        objective_callables=objective_callables,
        q=q,
        callable_format=callable_format,
        input_preprocessing_specs=input_preprocessing_specs,
        n_obj=n_obj,
        verbose=verbose,
    )

    res = pymoo_minimize(problem, algorithm, termination, verbose=verbose)

    if callable_format == "torch":
        # transform the result to the numeric domain
        x_opt = problem.domain_handler.transform_mixed_to_botorch_domain(
            [res.X]
        ).reshape((q, -1))
        f_opt = torch.from_numpy(res.F).to(**tkwargs)
    elif callable_format == "pandas":
        # transform the result to the original domain
        x_opt = problem.domain_handler.transform_to_experiments_per_individual(
            [res.X] if isinstance(res.X, dict) else res.X
        )
        f_opt = res.F

    return x_opt, f_opt
