import importlib.util
import re
import sys
import warnings
from copy import copy
from itertools import combinations
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import scipy.optimize as opt
from formulaic import Formula
from scipy import sparse
from scipy.optimize import NonlinearConstraint

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
from bofire.strategies.doe.objective_base import Objective
from bofire.strategies.doe.utils_categorical_discrete import (
    map_categorical_to_continuous,
)
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
        Au, bu, Au_neg = _row_subset(A, up), ub[up], -_row_subset(A, up)
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
        if constraint.features is None:  # ty: ignore[unresolved-attribute]
            raise ValueError(
                f"The features attribute of constraint {constraint} is not set, but has to be set.",
            )
        self.constraint_feature_indices = np.searchsorted(
            self.names,
            self.constraint.features,  # ty: ignore[unresolved-attribute]
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call constraint with flattened numpy array."""
        x = pd.DataFrame(
            x.reshape(len(x) // self.D, self.D), columns=self.names
        )  # ty: ignore[invalid-assignment]
        violation = self.constraint(x).to_numpy(copy=True)
        violation[np.abs(violation) < 0] = 0
        return violation

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense constraint Jacobian of shape ``(n_experiments, D * n_experiments)``.

        Used as the ``jac`` of the scipy ``NonlinearConstraint`` objects (the
        SLSQP path needs a dense Jacobian); the IPM backends use the sparse
        ``jacobian_coo`` directly. This is just ``jacobian_coo`` densified — the
        COO intermediate is negligible against the gradient evaluation and the
        (unavoidable, on the dense SLSQP path) dense allocation.
        """
        return self.jacobian_coo(x).toarray()

    def jacobian_coo(self, x: np.ndarray) -> sparse.coo_array:
        """Constraint Jacobian as a ``scipy.sparse.coo_array`` of shape
        ``(n_experiments, D * n_experiments)``.

        Block-diagonal: only the constraint's feature columns of each
        experiment's block are nonzero. Built directly as COO (no dense
        intermediate), so the sparse Jacobian carries through to the IPM
        solvers (cyipopt's / pounce's scipy interface) per iteration.
        """
        xdf = pd.DataFrame(
            x.reshape(len(x) // self.D, self.D), columns=self.names
        )  # ty: ignore[invalid-assignment]
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

    def hessian(self, x: np.ndarray, *args):
        """Call constraint hessian with flattened numpy array."""
        x = pd.DataFrame(
            x.reshape(len(x) // self.D, self.D), columns=self.names
        )  # ty: ignore[invalid-assignment]
        hessian_dict = self.constraint.hessian(x)

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


def _build_nchoosek_combined_patterns(
    domain: Domain,
) -> list[tuple[int, ...]]:
    """Collect combined deactivation patterns.

    Args:
        domain: Domain whose NChooseK constraints should be combined.

    Returns:
        A deduplicated list of inactive-index tuples.

    Raises:
        ValueError: When no valid combined patterns exist (contradictory
            constraints).
    """
    result = list(set(_get_nchoosek_combined_patterns(domain)))

    if not result and any(
        isinstance(c, NChooseKConstraint) for c in domain.constraints
    ):
        raise ValueError(
            "No valid combined NChooseK patterns exist. "
            "The overlapping constraints are contradictory."
        )

    return result


def nchoosek_constraints_as_bounds(
    domain: Domain,
    n_experiments: int,
) -> list:
    """Determines the bounds for the optimization problem that correspond to the NChooseK constraints of the domain.

    Args:
        domain (Domain): Domain to find the bounds for.
        n_experiments (int): number of experiments for the design to be determined.

    Returns:
        A list of tuples containing bounds that respect NChooseK constraints
        imposed onto the decision variables.

    """

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


# pounce's default L-BFGS + monotone-mu path stalls on the ill-conditioned
# D-optimality objective; mu_strategy=adaptive (+ acceptable termination) is the
# reliable limited-memory config. Overridable via ``optimizer_options``. (cyipopt's
# minimize_ipopt already defaults mu_strategy=adaptive internally.)
POUNCE_DOE_DEFAULTS = {
    "mu_strategy": "adaptive",
    "acceptable_tol": 1e-3,
    "acceptable_iter": 5,
}


def _densify_jacobians(constraints: List[dict]) -> List[dict]:
    """Return copies of the dict constraints with their (coo) ``jac`` densified.

    scipy SLSQP requires a dense constraint Jacobian (it raises on a sparse one),
    so the scipy path densifies; ipopt/pounce use the sparse coo jac directly.
    """

    def densify(jac):
        def wrapped(x, jac=jac):
            J = jac(x)
            return J.toarray() if sparse.issparse(J) else np.asarray(J)

        return wrapped

    return [{**c, "jac": densify(c["jac"])} for c in constraints]


def _scipy_options(optimizer_options: dict) -> dict:
    """Map Ipopt-style option names to scipy.optimize names."""
    options: dict = {}
    if optimizer_options:
        if "maxiter" in optimizer_options:
            options["maxiter"] = optimizer_options["maxiter"]
        if "max_iter" in optimizer_options:
            options["maxiter"] = optimizer_options["max_iter"]
        if "print_level" in optimizer_options:
            options["disp"] = bool(optimizer_options["print_level"])
    return options


def _resolve_optimizer(optimizer: str) -> str:
    """Fall back to scipy if the requested IPM backend is not installed."""
    if optimizer == "ipopt" and not CYIPOPT_AVAILABLE:
        warnings.warn(
            "optimizer='ipopt' but cyipopt is not installed; falling back to scipy.",
            stacklevel=2,
        )
        return "scipy"
    if optimizer == "pounce" and not POUNCE_AVAILABLE:
        warnings.warn(
            "optimizer='pounce' but pounce is not installed; falling back to scipy.",
            stacklevel=2,
        )
        return "scipy"
    return optimizer


def _minimize(
    objective_function: Objective,
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    constraints: Optional[List[dict]],
    optimizer: str = "ipopt",
    optimizer_options: Optional[dict] = None,
) -> np.ndarray:
    """Minimize the objective over the given bounds and constraints.

    ``constraints`` are the sparse coo-jac dicts from
    ``constraints_as_scipy_constraints``. Dispatches to one of three scipy-style
    backends:
    - box-only (no general constraints) → scipy ``L-BFGS-B`` regardless of
      ``optimizer`` (an IPM is overkill for a bound-only problem);
    - ``"scipy"`` → ``SLSQP`` on the constraints with their jac densified;
    - ``"ipopt"`` → ``cyipopt.minimize_ipopt`` with the sparse coo-jac dicts;
    - ``"pounce"`` → ``pounce.minimize`` with the sparse coo-jac dicts.
    ``"ipopt"``/``"pounce"`` fall back to scipy if the backend isn't installed.

    Args:
        objective_function: Objective providing ``evaluate``/``evaluate_jacobian``.
        x0: Initial guess (flattened design).
        bounds: Per-variable ``(lower, upper)`` bounds.
        constraints: dict constraints (``constraints_as_scipy_constraints`` output).
        optimizer: ``"ipopt"`` | ``"pounce"`` | ``"scipy"``.
        optimizer_options: options forwarded to the chosen backend.

    Returns:
        np.ndarray: The optimized design as a flattened numpy array.
    """
    optimizer_options = dict(optimizer_options or {})
    optimizer = _resolve_optimizer(optimizer)
    fun = objective_function.evaluate
    jac = objective_function.evaluate_jacobian

    # Box-only: no general constraints → bound-constrained → L-BFGS-B.
    if not constraints:
        return opt.minimize(
            fun=fun,
            x0=x0,
            jac=jac,
            bounds=bounds,
            method="L-BFGS-B",
            options=_scipy_options(optimizer_options),
        ).x

    if optimizer == "scipy":
        return opt.minimize(
            fun=fun,
            x0=x0,
            jac=jac,
            bounds=bounds,
            method="SLSQP",
            constraints=_densify_jacobians(constraints),  # SLSQP needs dense jacs
            options=_scipy_options(optimizer_options),
        ).x

    if optimizer == "ipopt":
        from cyipopt import minimize_ipopt

        return minimize_ipopt(
            fun,
            x0=x0,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options=optimizer_options,
        ).x
    if optimizer == "pounce":
        from pounce import minimize as pounce_minimize

        return pounce_minimize(
            fun,
            x0=x0,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            options={**POUNCE_DOE_DEFAULTS, **optimizer_options},
        ).x
    raise ValueError(f"unknown optimizer {optimizer!r}; use 'ipopt'|'pounce'|'scipy'")
