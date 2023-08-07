import sys
from itertools import combinations
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from formulaic import Formula
from scipy.optimize import LinearConstraint, NonlinearConstraint

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
)
from bofire.data_models.constraints.nonlinear import NonlinearInequalityConstraint
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    DiscreteInput,
    Output,
)
from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.strategies.api import (
    PolytopeSampler as PolytopeSamplerDataModel,
)
from bofire.strategies.doe.utils_features import (
    RelaxableBinaryInput,
    RelaxableDiscreteInput,
)
from bofire.strategies.samplers.polytope import PolytopeSampler


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
    N = len(model_formula) + 3

    sampler = PolytopeSampler(data_model=PolytopeSamplerDataModel(domain=domain))
    X = sampler.ask(N)
    # compute eigenvalues of information matrix
    A = model_formula.get_model_matrix(X)
    eigvals = np.abs(np.linalg.eigvalsh(A.T @ A))  # type: ignore

    return len(eigvals) - len(eigvals[eigvals > epsilon])


def constraints_as_scipy_constraints(
    domain: Domain,
    n_experiments: int,
    ignore_nchoosek: bool = True,
) -> List:
    """Formulates opti constraints as scipy constraints.

    Args:
        domain (Domain): Domain whose constraints should be formulated as scipy constraints.
        n_experiments (int): Number of instances of inputs for problem that are evaluated together.
        ingore_nchoosek (bool): NChooseK constraints are ignored if set to true. Defaults to True.

    Returns:
        A list of scipy constraints corresponding to the constraints of the given opti problem.
    """
    D = len(domain.inputs)

    # reformulate constraints
    constraints = []
    if len(domain.constraints) == 0:
        return constraints
    for c in domain.constraints:
        if isinstance(c, LinearEqualityConstraint):
            # write lower/upper bound as vector
            lb = np.ones(n_experiments) * (c.rhs / np.linalg.norm(c.coefficients))
            ub = np.ones(n_experiments) * (c.rhs / np.linalg.norm(c.coefficients))

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
            lb = np.zeros(n_experiments)
            ub = np.zeros(n_experiments)

            # define constraint evaluation (and gradient if provided)
            fun = ConstraintWrapper(
                constraint=c, domain=domain, n_experiments=n_experiments
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
                constraint=c, domain=domain, n_experiments=n_experiments
            )

            if c.jacobian_expression is not None:
                constraints.append(NonlinearConstraint(fun, lb, ub, jac=fun.jacobian))
            else:
                constraints.append(NonlinearConstraint(fun, lb, ub))

        elif isinstance(c, NChooseKConstraint):
            if ignore_nchoosek:
                pass
            else:
                # write upper/lower bound as vector
                lb = -np.inf * np.ones(n_experiments)
                ub = np.zeros(n_experiments)

                # define constraint evaluation (and gradient if provided)
                fun = ConstraintWrapper(
                    constraint=c, domain=domain, n_experiments=n_experiments
                )

                constraints.append(NonlinearConstraint(fun, lb, ub, jac=fun.jacobian))

        else:
            raise NotImplementedError(f"No implementation for this constraint: {c}")

    return constraints


class ConstraintWrapper:
    """Wrapper for nonlinear constraints."""

    def __init__(
        self, constraint: NonlinearConstraint, domain: Domain, n_experiments: int = 0
    ) -> None:
        """
        Args:
            constraint (Constraint): constraint to be called
            domain (Domain): Domain the constraint belongs to
        """
        self.constraint = constraint
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
        violation[np.abs(violation) < 0] = 0
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


def d_optimality(X: np.ndarray, delta=1e-9) -> float:
    """Compute ln(1/|X^T X|) for a model matrix X (smaller is better).
    The covariance of the estimated model parameters for $y = X beta + epsilon $is
    given by $Var(beta) ~ (X^T X)^{-1}$.
    The determinant |Var| quantifies the volume of the confidence ellipsoid which is to
    be minimized.
    """
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > delta]
    return np.sum(np.log(eigenvalues))


def a_optimality(X: np.ndarray, delta=1e-9) -> float:
    """Compute the A-optimality for a model matrix X (smaller is better).
    A-optimality is the sum of variances of the estimated model parameters, which is
    the trace of the covariance matrix $X.T @ X^-1$.

    F is symmetric positive definite, hence the trace of (X.T @ X)^-1 is equal to the
    the sum of inverse eigenvalues.
    """
    eigenvalues = np.linalg.eigvalsh(X.T @ X)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > delta]
    return np.sum(1.0 / eigenvalues)  # type: ignore


def g_optimality(X: np.ndarray, delta: float = 1e-9) -> float:
    """Compute the G-optimality for a model matrix X (smaller is better).
    G-optimality is the maximum entry in the diagonal of the hat matrix
    H = X (X.T X)^-1 X.T which relates to the maximum variance of the predicted values.
    """
    H = X @ np.linalg.inv(X.T @ X + delta * np.eye(len(X))) @ X.T
    return np.max(np.diag(H))  # type: ignore


def metrics(X: np.ndarray, delta: float = 1e-9) -> pd.Series:
    """Returns a series containing D-optimality, A-optimality and G-efficiency
    for a model matrix X

    Args:
        X (np.ndarray): model matrix for which the metrics are determined
        delta (float): cutoff value for eigenvalues of the information matrix. Default value is 1e-9.

    Returns:
        A pd.Series containing the values for the three metrics.
    """
    return pd.Series(
        {
            "D-optimality": d_optimality(X, delta),
            "A-optimality": a_optimality(X, delta),
            "G-optimality": g_optimality(X, delta),
        }
    )


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
        [p.bounds for p in domain.inputs.get(ContinuousInput)] * n_experiments  # type: ignore
    )

    if len(domain.constraints) > 0:
        for constraint in domain.constraints:
            if isinstance(constraint, NChooseKConstraint):
                n_inactive = len(constraint.features) - constraint.max_count

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
    else:
        pass

    # convert bounds to list of tuples
    bounds = [(b[0], b[1]) for b in bounds]

    return bounds


def discrete_to_relaxable_domain_mapper(
    domain: Domain,
) -> Tuple[Domain, List[List[RelaxableBinaryInput]]]:
    """Converts a domain with discrete and categorical inputs to a domain with relaxable inputs.

    Args:
        domain (Domain): Domain with discrete and categorical inputs.
    """

    # get all discrete and categorical inputs
    kept_inputs = domain.get_features(
        excludes=[CategoricalInput, DiscreteInput, Output]
    ).features
    discrete_inputs = domain.inputs.get(DiscreteInput).features
    categorical_inputs = domain.inputs.get(CategoricalInput).features

    # convert discrete inputs to continuous inputs
    relaxable_discrete_inputs = [
        RelaxableDiscreteInput(key=d_input.key, values=d_input.values)
        for d_input in discrete_inputs
    ]

    # convert categorical inputs to continuous inputs
    relaxable_categorical_inputs = []
    new_constraints = []
    categorical_groups = []
    for c_input in categorical_inputs:
        current_group_keys = list(c_input.categories)
        pick_1_constraint, group_vars = generate_mixture_constraints(current_group_keys)
        categorical_groups.append(group_vars)
        relaxable_categorical_inputs.extend(group_vars)
        new_constraints.append(pick_1_constraint)

    # create new domain with continuous inputs
    new_domain = Domain(
        inputs=kept_inputs + relaxable_discrete_inputs + relaxable_categorical_inputs,
        outputs=domain.outputs.features,
        constraints=domain.constraints.constraints + new_constraints,
    )

    return new_domain, categorical_groups


def NChooseKGroup_with_quantity(
    unique_group_identifier: str,
    keys: List[str],
    pick_at_least: int,
    pick_at_most: int,
    quantity_if_picked: Optional[
        Union[Tuple[float, float], List[Tuple[float, float]]]
    ] = None,
    combined_quantity_limit: Optional[float] = None,
    combined_quantity_is_equal_or_less_than: bool = False,
    use_non_relaxable_category_and_non_linear_constraint: bool = False,
) -> tuple[
    list[CategoricalInput] | list[RelaxableBinaryInput],
    list[ContinuousInput] | list[Any],
    list[LinearEqualityConstraint],
]:
    """
    helper function to generate an N choose K problem with categorical variables, with an option to connect each
    element of a category to a corresponding quantity of how much that category should be used.

    Args:
        unique_group_identifier (str): unique ID for the category/group which will be used to mark all variables
            containing to this group
        keys (List[str]): defines the names and the amount of the elements within the category
        pick_at_least (int): minimum number of elements to be picked from the category. >=0
        pick_at_most (int): maximum number of elements to be picked from the category. >=pick_at_least
        quantity_if_picked (Optional[Union[Tuple[float, float], List[Tuple[float, float]]]): If provided, specifies
            the lower and upper bound of the quantity, for each element in the category. List of bounds to specify the
            allowed quantity for each element separately or one single bound to set the same bounds for all elements.
        combined_quantity_limit (Optional[float]): If provided, sets an upper bound on what the sum of all the
            quantities of all elements should be
        combined_quantity_is_equal_or_less_than (bool): If True, the combined_quantity_limit describes the exact amount
            of the sum of all quantities. If False, it is a upper bound, i.e. the sum of the quantities can be lower.
            Default is False
        use_non_relaxable_category_and_non_linear_constraint (bool): Default is False.
            If False, RelaxableCategoricalInput is used in combination with LinearConstraints.
            If True, CategoricalInput used in combination with NonlinearConstraints, as CategoricalInput can not be
            used within LinearConstraints
    Returns:
        Either one CategoricalInput wrapped in a List or List of RelaxableBinaryInput describing the group,
        If quantities are provided, List of ContinuousInput describing the quantity of each element of the group
        otherwise empty List,
        List of either LinearConstraints or mix of Linear- and NonlinearConstraints, which enforce the quantities
        and group restrictions.
    """
    if quantity_if_picked is not None:
        if type(quantity_if_picked) is list and len(keys) != len(quantity_if_picked):
            raise ValueError(
                f"number of keys must be the same as corresponding quantities. Received {len(keys)} keys "
                f"and {len(quantity_if_picked)} quantities"
            )

        if type(quantity_if_picked) is list and True in [
            0 in q for q in quantity_if_picked
        ]:
            raise ValueError(
                "If an element out of the group is chosen, the quantity with which it is used must be "
                "larger than 0"
            )

    if pick_at_least > pick_at_most:
        raise ValueError(
            f"your upper bound to pick an element should be larger your lower bound. "
            f"Currently: pick_at_least {pick_at_least} > pick_at_most {pick_at_most}"
        )

    if pick_at_least < 0:
        raise ValueError(
            f"you should at least pick 0 elements. Currently  pick_at_least = {pick_at_least}"
        )

    if pick_at_most > len(keys):
        raise ValueError(
            f"you can not pick more elements than are available. "
            f"Received pick_at_most {pick_at_most} > number of keys {len(keys)}"
        )

    if "pick_none" in keys:
        raise ValueError("pick_none is not allowed as a key")

    if True in ["_" in k for k in keys]:
        raise ValueError('"_" is not allowed as an character in the keys')

    if quantity_if_picked is not None and type(quantity_if_picked) != list:
        quantity_if_picked = [quantity_if_picked for k in keys]

    quantity_var, all_new_constraints = [], []
    quantity_constraints_lb, quantity_constraints_ub = [], []
    max_quantity_constraint = None

    # creating possible combination of n choose k
    combined_keys_as_tuple = []
    if pick_at_most > 1:
        for i in range(max(2, pick_at_least), pick_at_most + 1):
            combined_keys_as_tuple.extend(list(combinations(keys, i)))
    if pick_at_least <= 1:
        combined_keys_as_tuple.extend([[k] for k in keys])

    combined_keys = ["_".join(w) for w in combined_keys_as_tuple]

    # generating the quantity variables and corresponding constraints
    if quantity_if_picked:
        (
            quantity_var,
            quantity_constraints_lb,
            quantity_constraints_ub,
            max_quantity_constraint,
        ) = _generate_quantity_var_constr(
            unique_group_identifier,
            keys,
            quantity_if_picked,
            combined_keys,
            combined_keys_as_tuple,
            use_non_relaxable_category_and_non_linear_constraint,
            combined_quantity_limit,
            combined_quantity_is_equal_or_less_than,
        )

    # allowing to pick none
    if pick_at_least == 0:
        combined_keys.append(unique_group_identifier + "_pick_none")

    # adding the new possible combinations to the list of keys
    keys = [unique_group_identifier + "_" + k for k in combined_keys]

    # choosing between CategoricalInput and RelaxableBinaryInput
    if use_non_relaxable_category_and_non_linear_constraint:
        category = [CategoricalInput(key=unique_group_identifier, categories=keys)]
        # if we use_legacy_class is true this constraint will be added by the discrete_to_relaxable_domain_mapper function
        pick_exactly_one_of_group_const = []
    else:
        category = [RelaxableBinaryInput(key=k) for k in keys]
        pick_exactly_one_of_group_const = [
            LinearEqualityConstraint(
                features=list(keys), coefficients=[1 for k in keys], rhs=1
            )
        ]

    all_new_constraints = (
        pick_exactly_one_of_group_const
        + quantity_constraints_lb
        + quantity_constraints_ub
    )
    if max_quantity_constraint is not None:
        all_new_constraints.append(max_quantity_constraint)
    return category, quantity_var, all_new_constraints


def _generate_quantity_var_constr(
    unique_group_identifier,
    keys,
    quantity_if_picked,
    combined_keys,
    combined_keys_as_tuple,
    use_non_relaxable_category_and_non_linear_constraint,
    combined_quantity_limit,
    combined_quantity_is_equal_or_less_than,
) -> tuple[
    list[ContinuousInput],
    list[NonlinearInequalityConstraint] | list[LinearInequalityConstraint],
    list[NonlinearInequalityConstraint] | list[LinearInequalityConstraint],
    LinearEqualityConstraint | LinearInequalityConstraint | None,
]:
    """
    Internal helper function just to create the quantity variables and the corresponding constraints.
    """
    quantity_var = [
        ContinuousInput(
            key=unique_group_identifier + "_" + k + "_quantity", bounds=(0, q[1])
        )
        for k, q in zip(keys, quantity_if_picked)
    ]

    all_quantity_features = []
    for k in keys:
        all_quantity_features.append(
            [
                unique_group_identifier + "_" + state_key
                for state_key, state_tuple in zip(combined_keys, combined_keys_as_tuple)
                if k in state_tuple
            ]
        )

    if use_non_relaxable_category_and_non_linear_constraint:
        quantity_constraints_lb = [
            NonlinearInequalityConstraint(
                expression="".join(
                    ["-" + unique_group_identifier + "_" + k + "_quantity"]
                    + [f" + {q[0]} * {key_c}" for key_c in combi]
                ),
                features=[unique_group_identifier + "_" + k + "_quantity"] + combi,
            )
            for combi, k, q in zip(all_quantity_features, keys, quantity_if_picked)
            if len(combi) >= 1
        ]

        quantity_constraints_ub = [
            NonlinearInequalityConstraint(
                expression="".join(
                    [unique_group_identifier + "_" + k + "_quantity"]
                    + [f" - {q[1]} * {key_c}" for key_c in combi]
                ),
                features=[unique_group_identifier + "_" + k + "_quantity"] + combi,
            )
            for combi, k, q in zip(all_quantity_features, keys, quantity_if_picked)
            if len(combi) >= 1
        ]
    else:
        quantity_constraints_lb = [
            LinearInequalityConstraint(
                features=[unique_group_identifier + "_" + k + "_quantity"] + combi,
                coefficients=[-1] + [q[0] for i in range(len(combi))],
                rhs=0,
            )
            for combi, k, q in zip(all_quantity_features, keys, quantity_if_picked)
            if len(combi) >= 1
        ]

        quantity_constraints_ub = [
            LinearInequalityConstraint(
                features=[unique_group_identifier + "_" + k + "_quantity"] + combi,
                coefficients=[1] + [-q[1] for i in range(len(combi))],
                rhs=0,
            )
            for combi, k, q in zip(all_quantity_features, keys, quantity_if_picked)
            if len(combi) >= 1
        ]

    max_quantity_constraint = None
    if combined_quantity_limit is not None:
        if combined_quantity_is_equal_or_less_than:
            max_quantity_constraint = LinearEqualityConstraint(
                features=[q.key for q in quantity_var],
                coefficients=[1 for q in quantity_var],
                rhs=combined_quantity_limit,
            )
        else:
            max_quantity_constraint = LinearInequalityConstraint(
                features=[q.key for q in quantity_var],
                coefficients=[1 for q in quantity_var],
                rhs=combined_quantity_limit,
            )

    return (
        quantity_var,
        quantity_constraints_lb,
        quantity_constraints_ub,
        max_quantity_constraint,
    )


def NChooseKGroup(
    variables: List[ContinuousInput],
    pick_at_least: int,
    pick_at_most: int,
    none_also_valid: bool,
) -> tuple[list[RelaxableBinaryInput], list[LinearConstraint],]:
    """
    helper function to generate an N choose K problem with categorical variables, with an option to connect each
    element of a category to a corresponding quantity of how much that category should be used.

    Args:
        variables (List[ContinuousInput]): variables to pick from
        pick_at_least (int): minimum number of elements to be picked from the category. >=0
        pick_at_most (int): maximum number of elements to be picked from the category. >=pick_at_least
        none_also_valid (bool): defines if also none of the elements can be picked
    Returns:
        List of RelaxableBinaryInput describing the group,
        List of either LinearConstraints, which enforce the quantities
        and group restrictions.
    """

    keys = [var.key for var in variables]
    if pick_at_least > pick_at_most:
        raise ValueError(
            f"your upper bound to pick an element should be larger your lower bound. "
            f"Currently: pick_at_least {pick_at_least} > pick_at_most {pick_at_most}"
        )

    if pick_at_least < 0:
        raise ValueError(
            f"you should at least pick 0 elements. Currently  pick_at_least = {pick_at_least}"
        )

    if pick_at_most > len(keys):
        raise ValueError(
            f"you can not pick more elements than are available. "
            f"Received pick_at_most {pick_at_most} > number of keys {len(keys)}"
        )

    if "pick_none" in keys:
        raise ValueError("pick_none is not allowed as a key")

    # creating possible combination of n choose k
    combined_keys_as_tuple = []
    if pick_at_most > 1:
        for i in range(max(2, pick_at_least), pick_at_most + 1):
            combined_keys_as_tuple.extend(list(combinations(keys, i)))
    if pick_at_least <= 1:
        combined_keys_as_tuple.extend([[k] for k in keys])

    combined_keys = ["_".join(w) for w in combined_keys_as_tuple]
    combined_keys = ["categorical_helper" + "_" + k for k in combined_keys]

    # generating the corresponding constraints
    valid_states = []
    for k in keys:
        valid_states.append(
            [
                state_key
                for state_key, state_tuple in zip(combined_keys, combined_keys_as_tuple)
                if k in state_tuple
            ]
        )

    quantity_constraints_lb = [
        LinearInequalityConstraint(
            features=[var.key] + combi,
            coefficients=[-1] + [var.lower_bound for i in range(len(combi))],
            rhs=0,
        )
        for combi, var in zip(valid_states, variables)
        if len(combi) >= 1
    ]

    quantity_constraints_ub = [
        LinearInequalityConstraint(
            features=[var.key] + combi,
            coefficients=[1] + [-var.upper_bound for i in range(len(combi))],
            rhs=0,
        )
        for combi, var in zip(valid_states, variables)
        if len(combi) >= 1
    ]

    # allowing to pick none
    if pick_at_least == 0 or none_also_valid:
        combined_keys.append("categorical_helper_pick_none_of_" + "".join(keys))

    # adding the new possible combinations to the list of keys
    keys = combined_keys

    category = [RelaxableBinaryInput(key=k) for k in keys]
    pick_exactly_one_of_group_const = [
        LinearEqualityConstraint(
            features=list(keys), coefficients=[1 for k in keys], rhs=1
        )
    ]

    all_new_constraints = (
        pick_exactly_one_of_group_const
        + quantity_constraints_lb
        + quantity_constraints_ub
    )

    return category, all_new_constraints


def generate_mixture_constraints(
    keys: List[str],
) -> Tuple[LinearEqualityConstraint, List[RelaxableBinaryInput]]:
    binary_vars = (RelaxableBinaryInput(key=x) for x in keys)

    mixture_constraint = LinearEqualityConstraint(
        features=keys, coefficients=[1 for x in range(len(keys))], rhs=1
    )

    return mixture_constraint, list(binary_vars)
