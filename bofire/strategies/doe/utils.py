import sys
import warnings
from itertools import combinations
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.optimize import LinearConstraint, NonlinearConstraint

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousOutput,
    MolecularInput,
)
from bofire.data_models.strategies.api import (
    PolytopeSampler as PolytopeSamplerDataModel,
)
from bofire.strategies.api import PolytopeSampler

CAT_TOL = 0.1
DISCRETE_TOL = 0.1


def get_formula_from_string(
    model_type: Union[str, Formula] = "linear",
    domain: Optional[Domain] = None,
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
    else:
        # linear model
        if model_type == "linear":
            formula = linear_formula(domain)

        # linear and interactions model
        elif model_type == "linear-and-quadratic":
            formula = linear_and_quadratic_formula(domain)

        # linear and quadratic model
        elif model_type == "linear-and-interactions":
            formula = linear_and_interactions_formula(domain)

        # fully quadratic model
        elif model_type == "fully-quadratic":
            formula = fully_quadratic_formula(domain)

        else:
            formula = model_type + "   "

    formula = Formula(formula[:-3])

    if rhs_only:
        if hasattr(formula, "rhs"):
            formula = formula.rhs

    # set recursion limit to old value
    sys.setrecursionlimit(recursion_limit)

    return formula


def linear_formula(
    domain: Optional[Domain],
) -> str:
    """Reformulates a string describing a linear-model or certain keywords as Formula objects.
        formula = model_type + "   "

    Args: domain (Domain): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.

    Returns:
        A string describing the model that was given as string or keyword.
    """
    assert (
        domain is not None
    ), "If the model is described by a keyword a domain must be provided"
    formula = "".join([input.key + " + " for input in domain.inputs])
    return formula


def linear_and_quadratic_formula(
    domain: Optional[Domain],
) -> str:
    """Reformulates a string describing a linear-and-quadratic model or certain keywords as Formula objects.

    Args: domain (Domain): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.

    Returns:
        A string describing the model that was given as string or keyword.
    """
    assert (
        domain is not None
    ), "If the model is described by a keyword a problem must be provided."
    formula = "".join([input.key + " + " for input in domain.inputs])
    formula += "".join(["{" + input.key + "**2} + " for input in domain.inputs])
    return formula


def linear_and_interactions_formula(
    domain: Optional[Domain],
) -> str:
    """Reformulates a string describing a linear-and-interactions model or certain keywords as Formula objects.

    Args: domain (Domain): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.

    Returns:
        A string describing the model that was given as string or keyword.
    """
    assert (
        domain is not None
    ), "If the model is described by a keyword a problem must be provided."
    formula = "".join([input.key + " + " for input in domain.inputs])
    for i in range(len(domain.inputs)):
        for j in range(i):
            formula += (
                domain.inputs.get_keys()[j] + ":" + domain.inputs.get_keys()[i] + " + "
            )
    return formula


def fully_quadratic_formula(
    domain: Optional[Domain],
) -> str:
    """Reformulates a string describing a fully-quadratic model or certain keywords as Formula objects.

    Args: domain (Domain): A problem context that nests necessary information on
        how to translate a problem to a formula. Contains a problem.

    Returns:
        A string describing the model that was given as string or keyword.
    """
    assert (
        domain is not None
    ), "If the model is described by a keyword a problem must be provided."
    formula = "".join([input.key + " + " for input in domain.inputs])
    for i in range(len(domain.inputs)):
        for j in range(i):
            formula += (
                domain.inputs.get_keys()[j] + ":" + domain.inputs.get_keys()[i] + " + "
            )
    formula += "".join(["{" + input.key + "**2} + " for input in domain.inputs])
    return formula


def n_zero_eigvals(
    domain: Domain, model_type: Union[str, Formula], epsilon=1e-7
) -> int:
    """Determine the number of eigenvalues of the information matrix that are necessarily zero because of
    equality constraints."""

    # sample points (fulfilling the constraints)
    model_formula = get_formula_from_string(
        model_type=model_type, rhs_only=True, domain=domain
    )
    N = len(model_formula.terms) + 3

    sampler = PolytopeSampler(data_model=PolytopeSamplerDataModel(domain=domain))
    X = sampler.ask(N)
    # compute eigenvalues of information matrix
    A = model_formula.get_model_matrix(X)
    eigvals = np.abs(np.linalg.eigvalsh(A.T @ A))  # type: ignore

    return len(eigvals) - len(eigvals[eigvals > epsilon])


def constraints_as_scipy_constraints(
    domain: Domain,
    n_experiments: int,
    tol: float = 1e-3,
) -> List:
    """Formulates opti constraints as scipy constraints. Ignores NchooseK constraints
    (these can be formulated as bounds).

    Args:
        domain (Domain): Domain whose constraints should be formulated as scipy constraints.
        n_experiments (int): Number of instances of inputs for problem that are evaluated together.
        tol (float): Tolerance for the computation of the constraint violation. Default value is 1e-3.

    Returns:
        A list of scipy constraints corresponding to the constraints of the given opti problem.
    """
    D = len(domain.inputs)

    # reformulate constraints
    constraints = []
    if len(domain.cnstrs) == 0:
        return constraints
    for c in domain.cnstrs:
        if isinstance(c, LinearEqualityConstraint):
            # write lower/upper bound as vector
            lb = np.ones(n_experiments) * (c.rhs / np.linalg.norm(c.coefficients) - tol)
            ub = np.ones(n_experiments) * (c.rhs / np.linalg.norm(c.coefficients) + tol)

            # write constraint as matrix
            lhs = {
                c.features[i]: c.coefficients[i] / np.linalg.norm(c.coefficients)
                for i in range(len(c.features))
            }
            row = np.zeros(D)
            for i, name in enumerate(domain.inputs.get_keys()):
                if name in lhs.keys():
                    row[i] = lhs[name]

            A = np.zeros(shape=(n_experiments, D * n_experiments))
            for i in range(n_experiments):
                A[i, i * D : (i + 1) * D] = row

            constraints.append(LinearConstraint(A, lb, ub))  # type: ignore

        elif isinstance(c, LinearInequalityConstraint):
            # write upper/lowe bound as vector
            lb = -np.inf * np.ones(n_experiments)
            ub = np.ones(n_experiments) * c.rhs / np.linalg.norm(c.coefficients)

            # write constraint as matrix
            lhs = {
                c.features[i]: c.coefficients[i] / np.linalg.norm(c.coefficients)
                for i in range(len(c.features))
            }
            row = np.zeros(D)
            for i, name in enumerate(domain.inputs.get_keys()):
                if name in lhs.keys():
                    row[i] = lhs[name]

            A = np.zeros(shape=(n_experiments, D * n_experiments))
            for i in range(n_experiments):
                A[i, i * D : (i + 1) * D] = row

            constraints.append(LinearConstraint(A, lb, ub))  # type: ignore

        elif isinstance(c, NonlinearEqualityConstraint):
            # write upper/lower bound as vector
            lb = np.zeros(n_experiments) - tol
            ub = np.zeros(n_experiments) + tol

            # define constraint evaluation (and gradient if provided)
            fun = ConstraintWrapper(
                constraint=c, domain=domain, n_experiments=n_experiments, tol=tol
            )

            if c.jacobian_expression is not None:
                constraints.append(NonlinearConstraint(fun, lb, ub, jac=fun.jacobian))
            else:
                constraints.append(NonlinearConstraint(fun, lb, ub))

        elif isinstance(c, NonlinearInequalityConstraint):
            # write upper/lower bound as vector
            lb = -np.inf * np.ones(n_experiments)
            ub = np.zeros(n_experiments)

            # define constraint evaluation (and gradient if provided)
            fun = ConstraintWrapper(
                constraint=c, domain=domain, n_experiments=n_experiments, tol=tol
            )

            if c.jacobian_expression is not None:
                constraints.append(NonlinearConstraint(fun, lb, ub, jac=fun.jacobian))
            else:
                constraints.append(NonlinearConstraint(fun, lb, ub))

        elif isinstance(c, NChooseKConstraint):
            pass

        else:
            raise NotImplementedError(f"No implementation for this constraint: {c}")

    return constraints


class ConstraintWrapper:
    """Wrapper for nonlinear constraints."""

    def __init__(
        self,
        constraint: NonlinearConstraint,
        domain: Domain,
        n_experiments: int = 0,
        tol: float = 1e-3,
    ) -> None:
        """
        Args:
            constraint (Constraint): constraint to be called
            domain (Domain): Domain the constraint belongs to
            tol (float): tolerance for constraint violation. Default value is 1e-3.
        """
        self.constraint = constraint
        self.tol = tol
        self.names = domain.inputs.get_keys()
        self.D = len(domain.inputs)
        self.n_experiments = n_experiments
        if constraint.features is None:
            raise ValueError(
                f"The features attribute of constraint {constraint} is not set, but has to be set."
            )
        self.constraint_feature_indices = np.searchsorted(
            self.names, self.constraint.features
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """call constraint with flattened numpy array."""
        x = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)  # type: ignore
        violation = self.constraint(x).to_numpy()
        violation[np.abs(violation) < self.tol] = 0
        return violation  # type: ignore

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """call constraint gradient with flattened numpy array."""
        x = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)
        gradient_compressed = self.constraint.jacobian(x).to_numpy()

        jacobian = np.zeros(shape=(self.n_experiments, self.D * self.n_experiments))
        rows = np.repeat(
            np.arange(self.n_experiments), len(self.constraint_feature_indices)
        )
        cols = np.repeat(
            self.D * np.arange(self.n_experiments), len(self.constraint_feature_indices)
        ).reshape((self.n_experiments, len(self.constraint_feature_indices)))
        cols = (cols + self.constraint_feature_indices).flatten()

        jacobian[rows, cols] = gradient_compressed.flatten()

        return jacobian


def d_optimality(X: np.ndarray, tol=1e-9) -> float:
    """Compute ln(1/|X^T X|) for a model matrix X (smaller is better).
    The covariance of the estimated model parameters for $y = X beta + epsilon $is
    given by $Var(beta) ~ (X^T X)^{-1}$.
    The determinant |Var| quantifies the volume of the confidence ellipsoid which is to
    be minimized.
    """
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > tol]
    return np.sum(np.log(eigenvalues))


def a_optimality(X: np.ndarray, tol=1e-9) -> float:
    """Compute the A-optimality for a model matrix X (smaller is better).
    A-optimality is the sum of variances of the estimated model parameters, which is
    the trace of the covariance matrix $X.T @ X^-1$.

    F is symmetric positive definite, hence the trace of (X.T @ X)^-1 is equal to the
    the sum of inverse eigenvalues
    """
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > tol]
    return np.sum(1.0 / eigenvalues)  # type: ignore


# type: ignore
def g_efficiency(
    X: np.ndarray,
    domain: Domain,
    delta: float = 1e-9,
    n_samples: int = int(1e4),
) -> float:
    """Compute the G-efficiency for a model matrix X.
    G-efficiency is proportional to p/(n*d) where p is the number of model terms,
    n is the number of runs and d is the maximum relative prediction variance over
    the set of runs.
    """

    # number of runs and model terms
    n, p = X.shape

    # take large sample from the design space
    sampler = PolytopeSampler(data_model=PolytopeSamplerDataModel(domain=domain))
    Y = sampler.ask(n_samples).to_numpy()

    # variance over set of runs
    D = Y @ np.linalg.inv(X.T @ X + delta * np.eye(p)) @ Y.T
    d = np.max(np.diag(D))
    if d:
        G_eff = float(100 * p / (n * d))
    else:
        G_eff = np.inf
    return G_eff


def metrics(
    X: np.ndarray,
    domain: Domain,
    tol: float = 1e-9,
    delta: float = 1e-9,
    n_samples: int = int(1e4),
) -> pd.Series:
    """Returns a series containing D-optimality, A-optimality and G-efficiency
    for a model matrix X

    Args:
        X (np.ndarray): model matrix for which the metrics are determined
        domain (Domain): domain definition containing the constraints of the design space.
        tol (float): cutoff value for eigenvalues of the information matrix in
            D- and A- optimality computation. Default value is 1e-9.
        delta (float): regularization parameter in G-efficiency computation.
            Default value is 1e-9
        n_samples (int): number of samples used to determine G-efficiency. Default value is 1e4.

    Returns:
        A pd.Series containing the values for the three metrics.
    """

    # X has to contain numerical values for metrics
    # thus transform to one-hot-encoded matrix
    # with properly transformed problem
    # try to determine G-efficiency
    try:
        g_eff = g_efficiency(
            X,
            domain,
            delta,
            n_samples,
        )

    except Exception:
        warnings.warn(
            "Sampling of points fulfilling this problem's constraints is not implemented. \
            G-efficiency can't be determined."
        )
        g_eff = 0

    return pd.Series(
        {
            "D-optimality": d_optimality(X, tol),
            "A-optimality": a_optimality(X, tol),
            "G-efficiency": g_eff,
        }
    )


def check_nchoosek_constraints_as_bounds(domain: Domain) -> None:
    """Checks if NChooseK constraints of domain can be formulated as bounds.

    Args:
        domain (Domain): Domain whose NChooseK constraints should be checked
    """
    # collect NChooseK constraints
    if len(domain.cnstrs) == 0:
        return

    nchoosek_constraints = []
    for c in domain.cnstrs:
        if isinstance(c, NChooseKConstraint):
            nchoosek_constraints.append(c)

    if len(nchoosek_constraints) == 0:
        return

    # check if the domains of all NCHooseK constraints are compatible to linearization
    for c in nchoosek_constraints:
        for name in np.unique(c.features):
            input = domain.inputs.get_by_key(name)
            if not (
                isinstance(input, CategoricalInput)
                or isinstance(input, ContinuousOutput)
                or isinstance(input, MolecularInput)
            ):
                if input.lower_bound > 0 or input.upper_bound < 0:
                    raise ValueError(
                        f"Constraint {c} cannot be formulated as bounds. 0 must be inside the \
                        domain of the affected decision variables."
                    )

    # check if the parameter names of two nchoose overlap
    for c in nchoosek_constraints:
        for _c in nchoosek_constraints:
            if c != _c:
                for name in c.features:
                    if name in _c.features:
                        raise ValueError(
                            f"Domain {domain} cannot be used for formulation as bounds. \
                            names attribute of NChooseK constraints must be pairwise disjoint."
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
        [
            (p.lower_bound, p.upper_bound)
            for p in domain.inputs
            if not (
                isinstance(p, CategoricalInput)
                or isinstance(p, ContinuousOutput)
                or isinstance(p, MolecularInput)
            )
        ]
        * n_experiments
    )

    if len(domain.cnstrs) > 0:
        for constraint in domain.cnstrs:
            if isinstance(constraint, NChooseKConstraint):
                n_inactive = len(constraint.features) - constraint.max_count

                # find indices of constraint.names in names
                ind = [
                    i
                    for i, p in enumerate(domain.inputs.get_keys())
                    if p in constraint.features
                ]

                # find and shuffle all combinations of elements of ind of length max_active
                ind = np.array([c for c in combinations(ind, r=n_inactive)])
                np.random.shuffle(ind)

                # set bounds to zero in each experiments for the variables that should be inactive
                for i in range(n_experiments):
                    ind_vanish = ind[i % len(ind)]
                    bounds[ind_vanish + i * len(domain.inputs), :] = [0, 0]
    else:
        pass

    # convert bounds to list of tuples
    bounds = [(b[0], b[1]) for b in bounds]

    return bounds
