import importlib.util
import re
import sys
import warnings
from copy import copy
from itertools import combinations
from typing import List, Optional, Tuple, Union, cast, get_args

import numpy as np
import pandas as pd
import scipy.optimize as opt
from formulaic import Formula
from scipy import sparse

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
from bofire.data_models.features.api import (
    CategoricalInput,
    DiscreteInput,
    NumericalInput,
)
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.strategies.doe import DEFAULT_DOE_OPTIMIZER, DoEOptimizer
from bofire.strategies.doe.objective_base import Objective
from bofire.strategies.doe.utils_categorical_discrete import (
    map_categorical_to_continuous,
)
from bofire.strategies.progress import AskProgressCallback, scipy_progress_callback
from bofire.strategies.random import RandomStrategy


CYIPOPT_AVAILABLE = importlib.util.find_spec("cyipopt") is not None
POUNCE_AVAILABLE = importlib.util.find_spec("pounce") is not None


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


def formula_str_to_fully_continuous(
    formula_str: str,
    inputs: Inputs,
) -> str:
    """Converts a formula with categorical variables to a formula with only continuous variables by identifying the categorical variables and replacing them with their one-hot encoded counterparts.
    E.g., if a categorical variable "color" has states "red", "blue", "green", the formula term "color" is replaced with "{color_red + color_blue}".
    """
    for cat_input in inputs.get([CategoricalInput]):
        _, categorical_one_hot_variabes, _ = map_categorical_to_continuous(
            categorical_inputs=[cat_input]
        )
        one_hot_terms = " + ".join(
            [var.key for var in categorical_one_hot_variabes[:-1]]
        )
        # Use word boundaries to match only complete variable names
        pattern = r"\b" + re.escape(cat_input.key) + r"\b"
        formula_str = re.sub(pattern, "(" + f"{one_hot_terms}" + ")", formula_str)

    formula = Formula(
        formula_str
    )  # formula casting for expansion of terms like (a+b)*(c+d)
    for _input in inputs.get([DiscreteInput]):
        for k in range(
            2, len(_input.values) + 1
        ):  # arbitrary upper bound on number of levels of discrete input
            if (len(_input.values) <= k) and (_input.key + f" ** {k}" in formula.root):
                warnings.warn(
                    f"Discrete input {_input.key} with {len(_input.values)} levels cannot represent a term of order {k} or higher.",
                    UserWarning,
                )
                break
    return str(formula)


def get_formula_from_string(
    model_type: str | Formula = "linear",
    inputs: Optional[Inputs] = None,
    rhs_only: bool = True,
) -> Formula:
    """Reformulates a string describing a model or certain keywords as Formula objects.

    Args:
        model_type (str or Formula): A formula containing all model terms.
        inputs (Inputs, optional): The inputs to be used in the formula. Defaults to None. If the model_type is a string describing a model type (e.g. "linear"), inputs must be provided to determine the formula. If the model_type is already a formula, inputs are not necessary and ignored if provided.
        rhs_only (bool): The function returns only the right hand side of the formula if set to True.

    Returns:
    A Formula object describing the model that was given as string or keyword.

    """
    # set maximum recursion depth to higher value
    recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(2000)

    if isinstance(model_type, Formula):
        return model_type

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
        if inputs is not None:
            if len(inputs.get([CategoricalInput])) > 0:
                model_type = formula_str_to_fully_continuous(
                    formula_str=model_type,
                    inputs=inputs,
                )
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
    _inputs = list(inputs.get([ContinuousInput])) + [
        input for input in inputs.get([DiscreteInput]) if len(input.values) > 2
    ]

    formula = "".join(["{" + input.key + "**2} + " for input in _inputs])
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
    # Need enough samples so the information matrix rank is determined by
    # structural constraints (e.g. equality constraints), not by sampling luck.
    # Discrete inputs with few levels are especially prone to degenerate samples
    # at small N.
    N = 5 * len(model_formula) + 3

    sampler = RandomStrategy(data_model=RandomStrategyDataModel(domain=domain))
    X = sampler.ask(N)
    # compute eigenvalues of information matrix
    A = model_formula.get_model_matrix(X)
    eigvals = np.abs(np.linalg.eigvalsh(A.T @ A))

    return len(eigvals) - len(eigvals[eigvals > epsilon])


def _row_subset(A: sparse.coo_array, mask: np.ndarray) -> sparse.coo_array:
    """Select rows of a COO matrix by a boolean mask, returning COO."""
    return A.tocsr()[mask].tocoo()


def _linear_constraint_dicts(A: sparse.coo_array, lb, ub) -> List[dict]:
    """Old-style dict constraints for a (two-sided) linear constraint ``lb <= A x <= ub``.

    scipy single-sided convention: ``eq`` → ``g == 0``, ``ineq`` → ``g >= 0``.
    Each ``jac`` returns a (constant) ``coo_array``.
    """
    m = A.shape[0]
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (m,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (m,))
    out: List[dict] = []
    eq = np.isclose(lb, ub) & np.isfinite(lb)
    if eq.any():  # A x = b
        Ae, be = _row_subset(A, eq), lb[eq]
        out.append(
            {
                "type": "eq",
                "fun": (lambda x, Ae=Ae, be=be: Ae @ x - be),
                "jac": (lambda x, Ae=Ae: Ae),
            }
        )
    up = np.isfinite(ub) & ~eq  # A x <= ub  ->  ub - A x >= 0
    if up.any():
        Au, bu = _row_subset(A, up), ub[up]
        Au_neg = -Au
        out.append(
            {
                "type": "ineq",
                "fun": (lambda x, Au=Au, bu=bu: bu - Au @ x),
                "jac": (lambda x, Au_neg=Au_neg: Au_neg),
            }
        )
    lo = np.isfinite(lb) & ~eq  # A x >= lb  ->  A x - lb >= 0
    if lo.any():
        Al, bl = _row_subset(A, lo), lb[lo]
        out.append(
            {
                "type": "ineq",
                "fun": (lambda x, Al=Al, bl=bl: Al @ x - bl),
                "jac": (lambda x, Al=Al: Al),
            }
        )
    return out


def _nonlinear_constraint_dicts(fun: "ConstraintWrapper", lb, ub) -> List[dict]:
    """Old-style dict constraint(s) for a nonlinear constraint, with a coo ``jac``
    from ``fun.jacobian_coo``. DoE nonlinear constraints are single-sided
    (equality ``g == b`` or upper-sided inequality ``g <= ub``)."""
    lb = np.atleast_1d(np.asarray(lb, dtype=float))
    ub = np.atleast_1d(np.asarray(ub, dtype=float))
    if np.all(np.isclose(lb, ub)):  # g(x) = b
        return [
            {
                "type": "eq",
                "fun": (lambda x, fun=fun, b=lb: np.atleast_1d(fun(x)) - b),
                "jac": (lambda x, fun=fun: fun.jacobian_coo(x)),
            }
        ]
    # g(x) <= ub  ->  ub - g(x) >= 0
    return [
        {
            "type": "ineq",
            "fun": (lambda x, fun=fun, ub=ub: ub - np.atleast_1d(fun(x))),
            "jac": (lambda x, fun=fun: -fun.jacobian_coo(x)),
        }
    ]


def constraints_as_scipy_constraints(
    domain: Domain,
    n_experiments: int,
    ignore_nchoosek: bool = True,
) -> List[dict]:
    """Formulate bofire constraints as scipy old-style **dict** constraints with a
    **sparse (coo) Jacobian**: ``{"type": "eq"|"ineq", "fun": ..., "jac": -> coo_array}``.

    This is the single representation used by all solver backends: it is consumed
    directly (sparse) by ``cyipopt.minimize_ipopt`` and ``pounce.minimize``, and
    densified on the fly for scipy/SLSQP (which needs a dense jac — see
    ``_densify_jacobians``). scipy's single-sided convention applies: ``eq`` means
    ``g(x) == 0`` and ``ineq`` means ``g(x) >= 0``.

    Args:
        domain (Domain): Domain whose constraints should be formulated.
        n_experiments (int): Number of experiments evaluated together.
        ignore_nchoosek (bool): NChooseK constraints are ignored if True. Default True.

    Returns:
        A list of dict constraints corresponding to the domain's constraints.
    """
    constraints: List[dict] = []
    for c in domain.constraints:
        if isinstance(
            c,
            (
                LinearEqualityConstraint,
                LinearInequalityConstraint,
                InterpointEqualityConstraint,
            ),
        ):
            A, lb, ub = get_constraint_function_and_bounds(c, domain, n_experiments)
            A_coo = sparse.coo_array(np.atleast_2d(np.asarray(A, dtype=float)))
            constraints.extend(_linear_constraint_dicts(A_coo, lb, ub))

        elif isinstance(
            c, (NonlinearEqualityConstraint, NonlinearInequalityConstraint)
        ):
            fun, lb, ub = get_constraint_function_and_bounds(c, domain, n_experiments)
            constraints.extend(_nonlinear_constraint_dicts(fun, lb, ub))

        elif isinstance(c, NChooseKConstraint):
            if not ignore_nchoosek:
                fun, lb, ub = get_constraint_function_and_bounds(
                    c, domain, n_experiments
                )
                constraints.extend(_nonlinear_constraint_dicts(fun, lb, ub))

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
            n_experiments=n_experiments,
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
            n_experiments=n_experiments,
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
        constraint: Constraint,
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
        if constraint.features is None:
            raise ValueError(
                f"The features attribute of constraint {constraint} is not set, but has to be set.",
            )
        self.constraint_feature_indices = np.searchsorted(
            self.names,
            self.constraint.features,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call constraint with flattened numpy array."""
        x = pd.DataFrame(
            x.reshape(len(x) // self.D, self.D), columns=self.names
        )  # ty: ignore[invalid-assignment]
        violation = self.constraint(x).to_numpy(copy=True)
        violation[np.abs(violation) < 0] = 0
        return violation

    def jacobian_coo(self, x: np.ndarray) -> sparse.coo_array:
        """Constraint Jacobian as a ``scipy.sparse.coo_array`` of shape
        ``(n_experiments, D * n_experiments)``.

        Block-diagonal: only the constraint's feature columns of each
        experiment's block are nonzero. The entries are the structural
        nonzeros, kept even where their value is 0, so the sparsity pattern
        stays fixed between iterations.
        """
        xdf = pd.DataFrame(x.reshape(len(x) // self.D, self.D), columns=self.names)
        gradient_compressed = self.constraint.jacobian(xdf).to_numpy()
        n_feat = len(self.constraint_feature_indices)
        rows = np.repeat(np.arange(self.n_experiments), n_feat)
        cols = np.repeat(self.D * np.arange(self.n_experiments), n_feat).reshape(
            (self.n_experiments, n_feat)
        )
        cols = (cols + self.constraint_feature_indices).flatten()
        return sparse.coo_array(
            (gradient_compressed.flatten(), (rows, cols)),
            shape=(self.n_experiments, self.D * self.n_experiments),
        )


def _constraint_patterns(
    all_keys: list[str],
    constraint: NChooseKConstraint,
) -> list[dict[int, bool]]:
    """Generate dicts for one constraint."""
    patterns_of_constraint = []
    n_features = len(constraint.features)
    ind = [i for i, key in enumerate(all_keys) if key in constraint.features]
    for k in range(max(constraint.min_count, 1), constraint.max_count + 1):
        n_inactive = n_features - k
        if n_inactive > 0:
            for combo in combinations(ind, r=n_inactive):
                inactive_set = set(combo)
                patterns_of_constraint.append(
                    {idx: (idx not in inactive_set) for idx in ind}
                )
        else:
            patterns_of_constraint.append({idx: True for idx in ind})
    return patterns_of_constraint


def _merge_two(
    patterns_of_constraint_a: list[dict[int, bool]],
    patterns_of_constraint_b: list[dict[int, bool]],
) -> list[dict[int, bool]]:
    """Merge two pattern lists, filtering conflicts.

    Uses a hash-join on shared feature indices: patterns from A are
    bucketed by their assignment to shared keys, so each pattern from B
    only needs to merge with the bucket that agrees on those keys.

    pattern_of_constraint_a: {feature_index: is_active} dict from constraint A
    pattern_of_constraint_b: {feature_index: is_active} dict from constraint B

    returns: merged {feature_index: is_active} dict if no conflicts, else None

    """
    # Identify shared feature indices between the two sides.
    shared = set(patterns_of_constraint_a[0].keys()).intersection(
        set(patterns_of_constraint_b[0].keys())
    )

    patterns_of_merged_constraints = []
    if shared:
        # Bucket patterns by their assignment on shared indices.
        # if activation_per_feature= {0: True, 1:True , 2: False} is a pattern and shared=[0,2], the key is (True, False)
        sorted_shared = sorted(shared)
        map_pattern_of_shared_to_global_pattern = {
            tuple(activation_per_feature[idx] for idx in sorted_shared): []
            for activation_per_feature in patterns_of_constraint_a
        }
        for activation_per_feature in patterns_of_constraint_a:
            pattern_of_shared = tuple(
                activation_per_feature[idx] for idx in sorted_shared
            )
            map_pattern_of_shared_to_global_pattern[pattern_of_shared].append(
                activation_per_feature
            )

        for activation_per_feature_b in patterns_of_constraint_b:
            pattern_of_shared = tuple(
                activation_per_feature_b[idx] for idx in sorted_shared
            )
            for activation_per_feature_a in map_pattern_of_shared_to_global_pattern.get(
                pattern_of_shared, ()
            ):
                merged = dict(activation_per_feature_a)
                merged.update(activation_per_feature_b)
                patterns_of_merged_constraints.append(merged)
    else:
        # No shared features — full cross-product.
        for activation_per_feature_b in patterns_of_constraint_b:
            for activation_per_feature_a in patterns_of_constraint_a:
                merged = dict(activation_per_feature_a)
                merged.update(activation_per_feature_b)
                patterns_of_merged_constraints.append(merged)

    return patterns_of_merged_constraints


def _get_nchoosek_combined_patterns(
    domain: Domain,
) -> list[tuple[int, ...]]:
    """Generate combined deactivation patterns across all NChooseK constraints.

    For each constraint, per-feature active/inactive patterns are generated
    for every allowed activity level.  When multiple constraints share
    features, patterns are merged incrementally (pairwise) rather than via
    a single giant Cartesian product, pruning inconsistent combinations
    early.

    Consistency: If a feature is shared by multiple constraints, it must be active in all or
    inactive in all to yield a valid combined pattern.

    Returns:
        Tuples containing the domain-level indices of features to deactivate
        (pin to 0) for one experiment slot.  Duplicates are possible when
        constraints are disjoint; the caller should deduplicate if needed.
    """
    nchoosek_constraints = [
        c for c in domain.constraints if isinstance(c, NChooseKConstraint)
    ]

    if not nchoosek_constraints:
        return []

    all_keys = domain.inputs.get_keys()

    # Incremental pairwise merge — each step only materialises a generator
    merged_patterns = _constraint_patterns(all_keys, nchoosek_constraints[0])
    for constraint in nchoosek_constraints[1:]:
        merged_patterns = _merge_two(
            merged_patterns, _constraint_patterns(all_keys, constraint)
        )

    allowed_constraint_patterns = []
    for merged in merged_patterns:
        allowed_constraint_patterns.append(
            tuple(sorted(idx for idx, active in merged.items() if not active))
        )
    return allowed_constraint_patterns


def _is_linearly_feasible(
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: np.ndarray,
) -> bool:
    """Check whether one point within ``bounds`` satisfies all linear rows.

    The rows are ``a_eq @ x == b_eq`` and ``a_ub @ x <= b_ub``; either set may
    have zero rows. Each row is first checked on its own against the interval
    it can attain within the bounds. A joint linear program is solved only when
    there are several rows, since individually attainable rows can still
    contradict one another.

    Args:
        a_eq: Equality coefficients, shape ``(n_eq, n_features)``.
        b_eq: Equality right-hand sides, shape ``(n_eq,)``.
        a_ub: Inequality coefficients, shape ``(n_ub, n_features)``.
        b_ub: Inequality upper bounds, shape ``(n_ub,)``.
        bounds: Lower and upper bound per feature, shape ``(n_features, 2)``.

    Returns:
        True if a feasible point exists, False if infeasibility is proven.

    Raises:
        RuntimeError: When the solver fails for a reason other than proven
            infeasibility.
    """
    matrix = np.vstack([a_eq, a_ub])
    rhs = np.concatenate([b_eq, b_ub])
    n_eq = len(b_eq)

    # Positive terms are smallest at lower bounds; negative terms at upper bounds.
    # Separating them gives each row's attainable minimum and maximum.
    positive = np.maximum(matrix, 0)
    negative = np.minimum(matrix, 0)
    minimum = positive @ bounds[:, 0] + negative @ bounds[:, 1]
    maximum = positive @ bounds[:, 1] + negative @ bounds[:, 0]
    # Inequalities need minimum <= rhs; equalities also need maximum >= rhs.
    tolerance = 1e-7  # HiGHS' default primal feasibility tolerance.
    if np.any(minimum > rhs + tolerance) or np.any(
        maximum[:n_eq] < rhs[:n_eq] - tolerance
    ):
        return False
    if len(rhs) == 1:
        return True

    # A zero objective asks only whether one point satisfies all rows together.
    result = opt.linprog(
        c=np.zeros(len(bounds)),
        A_eq=a_eq if n_eq else None,
        b_eq=b_eq if n_eq else None,
        A_ub=a_ub if len(b_ub) else None,
        b_ub=b_ub if len(b_ub) else None,
        bounds=bounds,
        method="highs",
    )
    if result.success:
        return True
    if result.status == 2:
        return False
    # Only proven infeasibility (status 2) justifies dropping a pattern;
    # numerical or solver failures must not silently shrink the search.
    raise RuntimeError(
        f"Linear feasibility check failed: {result.message} "
        f"(status {result.status})."
    )


def _filter_linearly_feasible_patterns(
    domain: Domain,
    patterns: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Keep patterns compatible with bounds and intrapoint linear constraints.

    Expects the fully continuous domain that ``nchoosek_constraints_as_bounds``
    guarantees. Feasibility checks use copies of the input bounds; the domain
    is not modified. Nonlinear and interpoint feasibility and the nonzero
    status of unpinned features are not established here.

    Raises:
        RuntimeError: When the linear feasibility solver fails for a pattern.
    """
    linear_constraints = domain.constraints.get(
        [LinearEqualityConstraint, LinearInequalityConstraint]
    )
    if not patterns or not linear_constraints:
        return patterns

    # Convert constraints once, in the same feature order used by the patterns.
    n_features = len(domain.inputs)
    equalities, equality_rhs, inequalities, inequality_rhs = [], [], [], []
    for constraint in linear_constraints:
        matrix, _, rhs = get_constraint_function_and_bounds(constraint, domain, 1)
        if isinstance(constraint, LinearEqualityConstraint):
            equalities.append(matrix[0])
            equality_rhs.append(rhs[0])
        else:
            inequalities.append(matrix[0])
            inequality_rhs.append(rhs[0])
    a_eq = np.array(equalities, dtype=float).reshape(-1, n_features)
    b_eq = np.array(equality_rhs, dtype=float)
    a_ub = np.array(inequalities, dtype=float).reshape(-1, n_features)
    b_ub = np.array(inequality_rhs, dtype=float)

    bounds = np.array(
        [feature.bounds for feature in domain.inputs.get(ContinuousInput)],
        dtype=float,
    )
    feasible = []
    for pattern in patterns:
        # A pattern lists inactive features: pin those to zero in a fresh copy.
        pattern_bounds = bounds.copy()
        pattern_bounds[list(pattern)] = 0
        try:
            if _is_linearly_feasible(a_eq, b_eq, a_ub, b_ub, pattern_bounds):
                feasible.append(pattern)
        except RuntimeError as error:
            raise RuntimeError(f"NChooseK pattern {pattern}: {error}") from error
    return feasible


def _build_nchoosek_combined_patterns(
    domain: Domain,
) -> list[tuple[int, ...]]:
    """Collect combined deactivation patterns compatible with linear constraints.

    Args:
        domain: Fully continuous domain whose NChooseK constraints should be
            combined; see ``nchoosek_constraints_as_bounds``.

    Returns:
        A deduplicated list of inactive-index tuples. The domain is not modified.

    Raises:
        ValueError: When no valid combined patterns exist (contradictory
            NChooseK constraints or infeasibility with bounds/linear constraints).
        RuntimeError: When the linear feasibility solver fails unexpectedly.
    """
    result = list(set(_get_nchoosek_combined_patterns(domain)))

    if not result and any(
        isinstance(c, NChooseKConstraint) for c in domain.constraints
    ):
        raise ValueError(
            "No valid combined NChooseK patterns exist. "
            "The overlapping constraints are contradictory."
        )

    feasible = _filter_linearly_feasible_patterns(domain, result)
    if result and not feasible:
        raise ValueError(
            "No combined NChooseK patterns are feasible under the input bounds "
            "and linear constraints."
        )
    return feasible


def nchoosek_constraints_as_bounds(
    domain: Domain,
    n_experiments: int,
) -> list:
    """Determines the bounds for the optimization problem that correspond to the NChooseK constraints of the domain.

    Internal to ``DoEStrategy``, which relaxes discrete and categorical inputs to
    continuous variables (``create_continuous_domain``) before calling this.
    Do not call it directly with a mixed domain.

    Args:
        domain (Domain): Fully continuous domain to find the bounds for.
        n_experiments (int): number of experiments for the design to be determined.

    Returns:
        A list of tuples containing bounds that respect NChooseK constraints
        imposed onto the decision variables. Patterns are screened for linear
        feasibility before being reused across experiment slots. The domain
        and its feature bounds are not modified.

    Raises:
        ValueError: When the domain contains non-continuous inputs.
    """
    # Bounds are built per continuous input, but NChooseK patterns index all
    # inputs. Only a fully continuous domain keeps the two aligned.
    if len(domain.inputs.get(ContinuousInput)) != len(domain.inputs):
        raise ValueError(
            "nchoosek_constraints_as_bounds expects a fully continuous domain. "
            "Discrete and categorical inputs must be relaxed to continuous "
            "variables first, as DoEStrategy does via create_continuous_domain; "
            "this function is not meant to be called directly with a mixed domain."
        )

    # bounds without NChooseK constraints
    bounds = np.array(
        [p.bounds for p in domain.inputs.get(ContinuousInput)] * n_experiments,
    )

    combined_patterns = _build_nchoosek_combined_patterns(domain)

    if combined_patterns:
        np.random.shuffle(combined_patterns)

        D = len(domain.inputs)
        for i in range(n_experiments):
            pattern = combined_patterns[i % len(combined_patterns)]
            for idx in pattern:
                bounds[idx + i * D, :] = [0, 0]
            if i % len(combined_patterns) == (len(combined_patterns) - 1):
                np.random.shuffle(combined_patterns)

    # convert bounds to list of tuples
    bounds = [(b[0], b[1]) for b in bounds]

    return bounds


# BoFire's settings for pounce's limited-memory (L-BFGS) path. An explicit
# ``mu_strategy`` also disables pounce's default-on fallback re-solve; the
# acceptable-level exit needs both a small error and a small relative change of
# the objective, so objectives of small magnitude (I-, E-optimality) are not cut
# short. Overridable via ``optimizer_options``.
POUNCE_DOE_DEFAULTS = {
    "mu_strategy": "adaptive",
    "acceptable_tol": 1e-3,
    "acceptable_iter": 5,
    "acceptable_obj_change_tol": 1e-6,
    # The acceptable exit may otherwise stop at a constraint violation up to
    # 1e-2 (Ipopt's default); keep it within the tolerance find_local_max
    # validates the design with.
    "acceptable_constr_viol_tol": 1e-4,
}

OPTIMIZERS = get_args(DoEOptimizer)


# Constraints are scipy dict constraints with a sparse ``jac``. Dicts are the only
# format that stays sparse for both interior-point backends:
# ``cyipopt.minimize_ipopt`` densifies ``LinearConstraint`` and
# ``NonlinearConstraint`` objects, and ``pounce.minimize`` accepts nonlinear
# constraints only as dicts. SLSQP rejects sparse dict Jacobians, hence
# ``_densify_jacobians``. Dropping SLSQP removes ``_densify_jacobians``; dropping
# cyipopt removes ``_ipopt_method`` and lets linear constraints become
# ``LinearConstraint`` objects. Nonlinear constraints stay dicts until pounce
# accepts ``NonlinearConstraint``.
def _densify_jacobians(constraints: List[dict]) -> List[dict]:
    """Return copies of the dict constraints with their sparse ``jac`` densified.

    SLSQP rejects sparse Jacobians in dict constraints.
    """

    def densify(jac):
        def wrapped(x, jac=jac):
            J = jac(x)
            return J.toarray() if sparse.issparse(J) else np.asarray(J)

        return wrapped

    return [{**c, "jac": densify(c["jac"])} for c in constraints]


def _scipy_options(optimizer_options: dict) -> dict:
    """Map Ipopt-style option names to scipy.optimize names. A scipy-style
    ``maxiter`` takes precedence over ``max_iter``, which ``find_local_max``
    always sets as a default."""
    options: dict = {}
    if optimizer_options:
        if "max_iter" in optimizer_options:
            options["maxiter"] = optimizer_options["max_iter"]
        if "maxiter" in optimizer_options:
            options["maxiter"] = optimizer_options["maxiter"]
        if "print_level" in optimizer_options:
            options["disp"] = bool(optimizer_options["print_level"])
    return options


def _check_optimizer(optimizer: str) -> None:
    """Raise if ``optimizer`` is unknown or its backend is not installed."""
    if optimizer not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}; choose one of {', '.join(OPTIMIZERS)}."
        )
    if optimizer == "pounce" and not POUNCE_AVAILABLE:
        raise ImportError(
            "optimizer='pounce' requires pounce. Install it via "
            "`pip install pounce-solver` (included in `bofire[optimization]`), or "
            "choose optimizer='scipy'."
        )
    if optimizer == "ipopt" and not CYIPOPT_AVAILABLE:
        raise ImportError(
            "optimizer='ipopt' requires cyipopt. Install it via "
            "`conda install -c conda-forge cyipopt`, or use the default "
            "optimizer='pounce'."
        )


def _ipopt_method(
    fun,
    x0,
    args=(),
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    callback=None,
    **options,
):
    """``cyipopt.minimize_ipopt`` with the signature of a
    ``scipy.optimize.minimize`` custom method, which passes options as keyword
    arguments rather than as one ``options`` dict."""
    from cyipopt import minimize_ipopt  # ty: ignore[unresolved-import]

    return minimize_ipopt(
        fun,
        x0,
        args=args,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options=options,
    )


def _solve(
    objective_function: Objective,
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    constraints: Optional[List[dict]],
    optimizer: str = DEFAULT_DOE_OPTIMIZER,
    optimizer_options: Optional[dict] = None,
    callback: Optional[AskProgressCallback] = None,
) -> opt.OptimizeResult:
    """Minimize the objective over the given bounds and constraints.

    Every backend runs through ``scipy.optimize.minimize`` with the same
    problem: the objective and its gradient, per-variable bounds, and the dict
    constraints from ``constraints_as_scipy_constraints``, whose ``jac`` returns
    a sparse ``coo_array``. scipy hands bounds and constraints unchanged to a
    callable ``method``, so the Jacobians stay sparse for pounce and cyipopt.

    - ``"pounce"``: ``method=pounce.minimize``, with ``POUNCE_DOE_DEFAULTS``
      underneath ``optimizer_options``.
    - ``"ipopt"``: ``method=_ipopt_method`` (cyipopt).
    - ``"scipy"``: ``method="SLSQP"`` with the Jacobians densified; Ipopt-style
      ``max_iter``/``print_level`` are translated.

    Args:
        objective_function: Objective providing ``evaluate``/``evaluate_jacobian``.
        x0: Initial guess (flattened design).
        bounds: Per-variable ``(lower, upper)`` bounds.
        constraints: dict constraints (``constraints_as_scipy_constraints`` output).
        optimizer: ``"pounce"``, ``"ipopt"`` or ``"scipy"``.
        optimizer_options: options forwarded to the chosen backend.
        callback: Called with an `AskProgressData` once per solver iteration
            (pounce also reports the starting point). Not supported with
            ``"ipopt"``, which warns and ignores it.

    Returns:
        The backend's ``OptimizeResult``; ``x`` is the flattened design.

    Raises:
        ValueError: If ``optimizer`` is unknown.
        ImportError: If the backend for ``optimizer`` is not installed.
    """
    _check_optimizer(optimizer)
    options = dict(optimizer_options or {})
    constraints = constraints or []
    if optimizer == "pounce":
        from pounce import minimize as method

        options = {**POUNCE_DOE_DEFAULTS, **options}
    elif optimizer == "ipopt":
        method = _ipopt_method
        if callback is not None:
            # cyipopt.minimize_ipopt raises on a callback and has no other
            # per-iteration hook.
            warnings.warn(
                "Progress is not reported with optimizer='ipopt'; use 'pounce' or "
                "'scipy' to receive progress callbacks.",
                UserWarning,
                stacklevel=2,
            )
            callback = None
    else:
        method = "SLSQP"
        constraints = _densify_jacobians(constraints)
        options = _scipy_options(options)

    return opt.minimize(
        objective_function.evaluate,
        x0,
        jac=objective_function.evaluate_jacobian,
        bounds=bounds,
        constraints=constraints,
        method=method,
        options=options,
        callback=None if callback is None else scipy_progress_callback(callback),
    )


def _minimize(
    objective_function: Objective,
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    constraints: Optional[List[dict]],
    optimizer: str = DEFAULT_DOE_OPTIMIZER,
    optimizer_options: Optional[dict] = None,
    callback: Optional[AskProgressCallback] = None,
) -> np.ndarray:
    """Like ``_solve``, but return only the optimized flattened design."""
    return _solve(
        objective_function,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        callback=callback,
    ).x
