import importlib.util
import sys
from copy import copy
from itertools import combinations
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import scipy.optimize as opt
from formulaic import Formula
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.optimize._minimize import standardize_constraints

from bofire.data_models.constraints.api import (
    Constraint,
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.features.api import CategoricalInput, NumericalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.strategies.doe.doe_problem import (
    FirstOrderDoEProblem,
    SecondOrderDoEProblem,
)
from bofire.strategies.doe.objective_base import Objective
from bofire.strategies.doe.utils_categorical_discrete import (
    map_categorical_to_continuous,
)
from bofire.strategies.random import RandomStrategy


CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None


def represent_categories_as_by_their_states(
    inputs: Inputs,
) -> Tuple[List[NumericalInput], List[ContinuousInput]]:
    all_but_one_categoricals = []
    if len(inputs.get([CategoricalInput])) > 0:
        inputs = copy(inputs)
        categorical_inputs = cast(
            list[CategoricalInput], inputs.get([CategoricalInput])
        )
        _, categorical_one_hot_variabes, _ = map_categorical_to_continuous(
            categorical_inputs=categorical_inputs
        )

        # enforce categoricals excluding each other
        all_but_one_categoricals = categorical_one_hot_variabes[:-1]
    numerical_inputs = cast(
        list[NumericalInput], list(inputs.get(excludes=[CategoricalInput]))
    )
    return numerical_inputs, all_but_one_categoricals


def get_formula_from_string(
    model_type: Union[str, Formula] = "linear",
    inputs: Optional[Inputs] = None,
    rhs_only: bool = True,
) -> Formula:
    """Reformulates a string describing a model or certain keywords as Formula objects.

    Args:
        model_type (str or Formula): A formula containing all model terms.
        domain (Domain): A domain that nests necessary information on
        how to translate a problem to a formula. Contains a problem.
        rhs_only (bool): The function returns only the right hand side of the formula if set to True.

    Returns:
    A Formula object describing the model that was given as string or keyword.

    """
    # set maximum recursion depth to higher value
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(2000)

    if isinstance(model_type, Formula):
        return model_type
        # build model if a keyword and a problem are given.
    # linear model#

    if model_type in [
        "linear",
        "linear-and-quadratic",
        "linear-and-interactions",
        "fully-quadratic",
    ]:
        if inputs is None:
            raise AssertionError(
                "Inputs must be provided if only a model type is given.",
            )
        continuous_inputs, categorical_inputs = represent_categories_as_by_their_states(
            inputs=inputs
        )
        if model_type == "linear":
            formula = linear_terms(
                inputs=Inputs(features=continuous_inputs + categorical_inputs)
            )

        # linear and interactions model
        elif model_type == "linear-and-quadratic":
            formula = linear_terms(
                inputs=Inputs(features=continuous_inputs + categorical_inputs)
            ) + quadratic_terms(inputs=Inputs(features=continuous_inputs))

        # linear and quadratic model
        elif model_type == "linear-and-interactions":
            formula = linear_terms(
                inputs=Inputs(features=continuous_inputs + categorical_inputs)
            ) + interactions_terms(
                continuous_inputs=Inputs(features=continuous_inputs),
                categorical_inputs=Inputs(features=categorical_inputs),
            )

        # fully quadratic model
        elif model_type == "fully-quadratic":
            formula = (
                linear_terms(
                    inputs=Inputs(features=continuous_inputs + categorical_inputs)
                )
                + interactions_terms(
                    continuous_inputs=Inputs(features=continuous_inputs),
                    categorical_inputs=Inputs(features=categorical_inputs),
                )
                + quadratic_terms(inputs=Inputs(features=continuous_inputs))
            )

        else:
            raise ValueError(
                f"Model type {model_type} is not supported. Supported model types are: "
                f"linear, linear-and-quadratic, linear-and-interactions, fully-quadratic.",
            )

    else:
        formula = model_type + "   "

    formula = Formula(formula[:-3])

    if rhs_only:
        if hasattr(formula, "rhs"):
            formula = formula.rhs

    # set recursion limit to old value
    sys.setrecursionlimit(recursion_limit)

    return formula


def convert_formula_to_string(
    domain: Domain,
    formula: Formula,
) -> str:
    """Converts a formula to a string.

    Args:
        domain (Domain): The domain that contain information about the input names.
        formula (Formula): A formula object that should be converted to a string. If
        the formula has both a left and right hand side, only the right hand side is
        considered.

    Returns:
        A string representation of the formula that can be evaluated using pytorch.

    """
    if hasattr(formula, "rhs"):
        formula = formula.rhs

    term_list = [str(term) for term in list(formula)]

    term_list_string = "torch.vstack(["
    for term in term_list:
        if term == "1":
            term_list_string += f"torch.ones_like({domain.inputs.get_keys()[0]}), "
        else:
            term_list_string += term.replace(":", "*") + ", "
    term_list_string += "]).T"

    return term_list_string


def linear_terms(
    inputs: Inputs,
) -> str:
    """Reformulates a string describing a linear-model or certain keywords as Formula objects.
        formula = model_type + "   "

    Args: inputs (Inputs): The inputs that should be used to build the linear model.

    Returns:
        A string describing the model that was given as string or keyword.

    """
    formula = "".join([input.key + " + " for input in inputs])
    return formula


def quadratic_terms(
    inputs: Inputs,
) -> str:
    """Reformulates a string describing a linear-and-quadratic model or certain keywords as Formula objects.

    Args: inputs (Inputs): The inputs that should be used to build the linear and quadratic model.

    Returns:
        A string describing the model that was given as string or keyword.

    """

    formula = "".join(["{" + input.key + "**2} + " for input in inputs])
    return formula


def interactions_terms(
    continuous_inputs: Inputs,
    categorical_inputs: Inputs,
) -> str:
    """Reformulates a string describing a linear-and-interactions model or certain keywords as Formula objects.

    Args: inputs (Inputs): The inputs that should be used to build the linear and interactions model.

    Returns:
        A string describing the model that was given as string or keyword.

    """
    inputs = continuous_inputs + categorical_inputs
    formula = ""
    for c in combinations(range(len(inputs)), 2):
        if not (
            (inputs.get_keys()[c[0]] in categorical_inputs.get_keys())
            and (inputs.get_keys()[c[1]] in categorical_inputs.get_keys())
        ):
            formula += inputs.get_keys()[c[0]] + ":" + inputs.get_keys()[c[1]] + " + "
    return formula


def n_zero_eigvals(
    domain: Domain,
    model_type: Union[str, Formula],
    epsilon=1e-7,
) -> int:
    """Determine the number of eigenvalues of the information matrix that are necessarily zero because of
    equality constraints.
    """
    # sample points (fulfilling the constraints)
    model_formula = get_formula_from_string(
        model_type=model_type,
        rhs_only=True,
        inputs=domain.inputs,
    )
    N = len(model_formula) + 3

    sampler = RandomStrategy(data_model=RandomStrategyDataModel(domain=domain))
    X = sampler.ask(N)
    # compute eigenvalues of information matrix
    A = model_formula.get_model_matrix(X)
    eigvals = np.abs(np.linalg.eigvalsh(A.T @ A))

    return len(eigvals) - len(eigvals[eigvals > epsilon])


def constraints_as_scipy_constraints(
    domain: Domain,
    n_experiments: int,
    ignore_nchoosek: bool = True,
) -> List:
    """Formulates bofire constraints as scipy constraints.

    Args:
        domain (Domain): Domain whose constraints should be formulated as scipy constraints.
        n_experiments (int): Number of instances of inputs for problem that are evaluated together.
        ingore_nchoosek (bool): NChooseK constraints are ignored if set to true. Defaults to True.

    Returns:
        A list of scipy constraints corresponding to the constraints of the given opti problem.

    """
    # reformulate constraints
    constraints = []
    if len(domain.constraints) == 0:
        return constraints
    for c in domain.constraints:
        if isinstance(c, LinearEqualityConstraint) or isinstance(
            c,
            LinearInequalityConstraint,
        ):
            A, lb, ub = get_constraint_function_and_bounds(c, domain, n_experiments)
            constraints.append(LinearConstraint(A, lb, ub))

        elif isinstance(c, NonlinearEqualityConstraint) or isinstance(
            c,
            NonlinearInequalityConstraint,
        ):
            fun, lb, ub = get_constraint_function_and_bounds(c, domain, n_experiments)
            constraints.append(
                NonlinearConstraint(fun, lb, ub, jac=fun.jacobian, hess=fun.hessian)
            )

        elif isinstance(c, NChooseKConstraint):
            if ignore_nchoosek:
                pass
            else:
                fun, lb, ub = get_constraint_function_and_bounds(
                    c,
                    domain,
                    n_experiments,
                )
                constraints.append(
                    NonlinearConstraint(fun, lb, ub, jac=fun.jacobian, hess=fun.hessian)
                )

        elif isinstance(c, InterpointEqualityConstraint):
            A, lb, ub = get_constraint_function_and_bounds(c, domain, n_experiments)
            constraints.append(LinearConstraint(A, lb, ub))

        else:
            raise NotImplementedError(f"No implementation for this constraint: {c}")

    return constraints


def get_constraint_function_and_bounds(
    c: Constraint,
    domain: Domain,
    n_experiments: int,
) -> List:
    """Returns the function definition and bounds for a given constraint and domain.

    Args:
        c (Constraint): Constraint for which the constraint matrix should be determined.
        domain (Domain): Domain for which the constraint matrix should be determined.
        n_experiments (int): Number of experiments for which the constraint matrix should be determined.

    Returns:
        A list containing the constraint defining function and the lower and upper bounds.

    """
    D = len(domain.inputs)

    if isinstance(c, LinearEqualityConstraint) or isinstance(
        c,
        LinearInequalityConstraint,
    ):
        # write constraint as matrix
        lhs = {
            c.features[i]: c.coefficients[i] / np.linalg.norm(c.coefficients)
            for i in range(len(c.features))
        }
        row = np.zeros(D)
        for i, name in enumerate(domain.inputs.get_keys()):
            if name in lhs:
                row[i] = lhs[name]

        A = np.zeros(shape=(n_experiments, D * n_experiments))
        for i in range(n_experiments):
            A[i, i * D : (i + 1) * D] = row

        # write upper/lower bound as vector
        lb = -np.inf * np.ones(n_experiments)
        ub = np.ones(n_experiments) * (c.rhs / np.linalg.norm(c.coefficients))
        if isinstance(c, LinearEqualityConstraint):
            lb = np.ones(n_experiments) * (c.rhs / np.linalg.norm(c.coefficients))

        return [A, lb, ub]

    if isinstance(c, NonlinearEqualityConstraint) or isinstance(
        c,
        NonlinearInequalityConstraint,
    ):
        # define constraint evaluation (and gradient if provided)
        fun = ConstraintWrapper(
            constraint=c,
            domain=domain,
            n_experiments=n_experiments,  # type: ignore
        )

        # write upper/lower bound as vector
        lb = -np.inf * np.ones(n_experiments)
        ub = np.zeros(n_experiments)
        if isinstance(c, NonlinearEqualityConstraint):
            lb = np.zeros(n_experiments)

        return [fun, lb, ub]

    if isinstance(c, NChooseKConstraint):
        # define constraint evaluation (and gradient if provided)
        fun = ConstraintWrapper(
            constraint=c,
            domain=domain,
            n_experiments=n_experiments,  # type: ignore
        )

        # write upper/lower bound as vector
        lb = -np.inf * np.ones(n_experiments)
        ub = np.zeros(n_experiments)

        return [fun, lb, ub]

    if isinstance(c, InterpointEqualityConstraint):
        # write lower/upper bound as vector
        multiplicity = c.multiplicity or len(domain.inputs)
        n_batches = int(np.ceil(n_experiments / multiplicity))
        lb = np.zeros(n_batches * (multiplicity - 1))
        ub = np.zeros(n_batches * (multiplicity - 1))

        # write constraint as matrix
        feature_idx = 0
        if c.feature not in domain.inputs.get_keys():
            raise ValueError(f"Feature {c.feature} is not part of the domain {domain}.")
        for i, name in enumerate(domain.inputs.get_keys()):
            if name == c.feature:
                feature_idx = i

        A = np.zeros(shape=(n_batches * (multiplicity - 1), D * n_experiments))
        for batch in range(n_batches):
            for i in range(multiplicity - 1):
                if batch * multiplicity + i + 2 <= n_experiments:
                    A[
                        batch * (multiplicity - 1) + i,
                        batch * multiplicity * D + feature_idx,
                    ] = 1.0
                    A[
                        batch * (multiplicity - 1) + i,
                        (batch * multiplicity + i + 1) * D + feature_idx,
                    ] = -1.0

        # remove overflow in last batch
        if (n_experiments % multiplicity) != 0:
            n_overflow = multiplicity - (n_experiments % multiplicity)
            A = A[:-n_overflow, :]
            lb = lb[:-n_overflow]
            ub = ub[:-n_overflow]

        return [A, lb, ub]

    raise NotImplementedError(f"No implementation for this constraint: {c}")


class ConstraintWrapper:
    """Wrapper for nonlinear constraints."""

    def __init__(
        self,
        constraint: NonlinearConstraint,
        domain: Domain,
        n_experiments: int = 0,
    ) -> None:
        """Args:
        constraint (Constraint): constraint to be called
        domain (Domain): Domain the constraint belongs to
        """
        self.constraint = constraint
        self.names = domain.inputs.get_keys()
        self.D = len(domain.inputs)
        self.n_experiments = n_experiments
        if constraint.features is None:  # type: ignore
            raise ValueError(
                f"The features attribute of constraint {constraint} is not set, but has to be set.",
            )
        self.constraint_feature_indices = np.searchsorted(
            self.names,
            self.constraint.features,  # type: ignore
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call constraint with flattened numpy array."""
        x = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)  # type: ignore
        violation = self.constraint(x).to_numpy()  # type: ignore
        violation[np.abs(violation) < 0] = 0
        return violation

    def jacobian(self, x: np.ndarray, sparse: bool = False) -> np.ndarray:
        """Call constraint gradient with flattened numpy array.  If sparse is set to True, the output is a vector containing the entries of the sparse matrix representation of the jacobian."""
        x = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)  # type: ignore
        gradient_compressed = self.constraint.jacobian(x).to_numpy()  # type: ignore

        cols = np.repeat(
            self.D * np.arange(self.n_experiments),
            len(self.constraint_feature_indices),
        ).reshape((self.n_experiments, len(self.constraint_feature_indices)))
        cols = (cols + self.constraint_feature_indices).flatten()

        if sparse:
            jacobian = np.zeros(shape=(self.D * self.n_experiments))
            jacobian[cols] = gradient_compressed.flatten()
            return jacobian

        rows = np.repeat(
            np.arange(self.n_experiments),
            len(self.constraint_feature_indices),
        )
        jacobian = np.zeros(shape=(self.n_experiments, self.D * self.n_experiments))
        jacobian[rows, cols] = gradient_compressed.flatten()

        return jacobian

    def hessian(self, x: np.ndarray, *args):
        """Call constraint hessian with flattened numpy array."""
        x = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)  # type: ignore
        hessian_dict = self.constraint.hessian(x)  # type: ignore

        hessian = np.zeros(
            shape=(self.D * self.n_experiments, self.D * self.n_experiments)
        )

        cols, rows = np.meshgrid(
            self.constraint_feature_indices,
            self.constraint_feature_indices,
        )
        for i, idx in enumerate(hessian_dict.keys()):
            hessian[i * self.D + cols, i * self.D + rows] = hessian_dict[idx]

        return hessian


def check_nchoosek_constraints_as_bounds(domain: Domain) -> None:
    """Checks if NChooseK constraints of domain can be formulated as bounds.

    Args:
        domain (Domain): Domain whose NChooseK constraints should be checked

    """
    # collect NChooseK constraints
    if len(domain.constraints) == 0:
        return

    nchoosek_constraints = []
    for c in domain.constraints:
        if isinstance(c, NChooseKConstraint):
            nchoosek_constraints.append(c)

    if len(nchoosek_constraints) == 0:
        return

    # check if the domains of all NChooseK constraints are compatible to linearization
    for c in nchoosek_constraints:
        for name in np.unique(c.features):
            input = domain.inputs.get_by_key(name)
            if input.bounds[0] > 0 or input.bounds[1] < 0:  # type: ignore
                raise ValueError(
                    f"Constraint {c} cannot be formulated as bounds. 0 must be inside the \
                    domain of the affected decision variables.",
                )

    # check if the parameter names of two nchoose overlap
    for c in nchoosek_constraints:
        for _c in nchoosek_constraints:
            if c != _c:
                for name in c.features:
                    if name in _c.features:
                        raise ValueError(
                            f"Domain {domain} cannot be used for formulation as bounds. \
                            names attribute of NChooseK constraints must be pairwise disjoint.",
                        )


def nchoosek_constraints_as_bounds(
    domain: Domain,
    n_experiments: int,
) -> List:
    """Determines the box bounds for the decision variables

    Args:
        domain (Domain): Domain to find the bounds for.
        n_experiments (int): number of experiments for the design to be determined.

    Returns:
        A list of tuples containing bounds that respect NChooseK constraint imposed
        onto the decision variables.

    """
    check_nchoosek_constraints_as_bounds(domain)

    # bounds without NChooseK constraints
    bounds = np.array(
        [p.bounds for p in domain.inputs.get(ContinuousInput)] * n_experiments,  # type: ignore
    )

    if len(domain.constraints) > 0:
        for constraint in domain.constraints:
            if isinstance(constraint, NChooseKConstraint):
                n_inactive = len(constraint.features) - constraint.max_count
                if n_inactive > 0:
                    # find indices of constraint.names in names
                    ind = [
                        i
                        for i, p in enumerate(domain.inputs.get_keys())
                        if p in constraint.features
                    ]

                    # find and shuffle all combinations of elements of ind of length max_active
                    ind = np.array(list(combinations(ind, r=n_inactive)))
                    np.random.shuffle(ind)

                    # set bounds to zero in each experiments for the variables that should be inactive
                    for i in range(n_experiments):
                        ind_vanish = ind[i % len(ind)]
                        bounds[ind_vanish + i * len(domain.inputs), :] = [0, 0]
                        if i % len(ind) == len(ind) - 1:
                            np.random.shuffle(ind)
    else:
        pass

    # convert bounds to list of tuples
    bounds = [(b[0], b[1]) for b in bounds]

    return bounds


def _minimize(
    objective_function: Objective,
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    constraints: Optional[List[Union[NonlinearConstraint, LinearConstraint]]],
    ipopt_options: dict,
    use_hessian: bool,
    use_cyipopt: Optional[bool] = CYIPOPT_AVAILABLE,
) -> np.ndarray:
    """Minimize the objective function using the given constraints and bounds.
    Uses Ipopt if available, otherwise uses SLSQP.

    Args:
        objective_function (Objective): Objective function to minimize.
        x0 (np.ndarray): Initial guess for the minimization.
        bounds (List[Tuple[float, float]]): Bounds for the decision variables.
        constraints (Optional[List[Union[NonlinearConstraint, LinearConstraint]]]): Constraints for the optimization problem.
        ipopt_options (dict): Options for Ipopt solver. If Ipopt is not available, only the fields "max_iter" and "print_level" of this argument are used.
        use_hessian (bool): Use hessian if set to True.
        use_cyipopt (bool): Use cyipopt if set to True. Defaults to true if cyipopt is available.

    Returns:
        np.ndarray: The optimized design as flattened numpy array.
    """
    if use_cyipopt is None:
        use_cyipopt = CYIPOPT_AVAILABLE

    if use_cyipopt:
        if use_hessian:
            problem = SecondOrderDoEProblem(
                doe_objective=objective_function,
                bounds=bounds,
                constraints=constraints,
            )
        else:
            problem = FirstOrderDoEProblem(
                doe_objective=objective_function,
                bounds=bounds,
                constraints=constraints,
            )
        for key in ipopt_options.keys():
            problem.add_option(key, ipopt_options[key])

        x, info = problem.solve(x0)
        return x
    else:
        options = {}
        if "max_iter" in ipopt_options.keys():
            options["maxiter"] = ipopt_options["max_iter"]
        if "print_level" in ipopt_options.keys():
            options["disp"] = ipopt_options["print_level"]
        result = opt.minimize(
            fun=objective_function.evaluate,
            x0=x0,
            bounds=bounds,
            options=options,
            constraints=standardize_constraints(constraints, x0, "SLSQP"),
            jac=objective_function.evaluate_jacobian,
            hess=objective_function.evaluate_hessian if use_hessian else None,
        )
        return result.x
