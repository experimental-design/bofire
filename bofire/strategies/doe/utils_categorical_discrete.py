import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bofire.data_models.constraints.linear import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.constraints.nchoosek import NChooseKConstraint
from bofire.data_models.constraints.nonlinear import NonlinearInequalityConstraint
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.categorical import CategoricalInput
from bofire.data_models.features.continuous import ContinuousInput
from bofire.data_models.features.discrete import DiscreteInput
from bofire.data_models.features.feature import Feature, Output, TDiscreteVals


def discrete_to_relaxable_domain_mapper(
    domain: Domain,
) -> Tuple[
    Domain,
    List[List[ContinuousInput]],
    Dict[str, Tuple[ContinuousInput, TDiscreteVals]],
]:
    """Converts a domain with discrete and categorical inputs to a domain with relaxable inputs.

    Args:
        domain (Domain): Domain with discrete and categorical inputs.
    """

    # get all discrete and categorical inputs
    kept_inputs = domain.get_features(
        excludes=[CategoricalInput, DiscreteInput, Output]
    ).features
    discrete_inputs = domain.inputs.get(DiscreteInput)
    categorical_inputs = domain.inputs.get(CategoricalInput)

    # convert discrete inputs to continuous inputs
    relaxable_discrete_inputs = {
        d_input.key: (  # type: ignore
            ContinuousInput(key=d_input.key, bounds=(min(d_input.values), max(d_input.values))),  # type: ignore
            d_input.values,  # type: ignore
        )  # type: ignore
        for d_input in discrete_inputs
    }

    # convert categorical inputs to continuous inputs
    relaxable_categorical_inputs = []
    new_constraints = []
    categorical_groups: List[List[ContinuousInput]] = []
    for c_input in categorical_inputs:
        current_group_keys = list(c_input.categories)  # type: ignore
        pick_1_constraint, group_vars = generate_mixture_constraints(current_group_keys)
        categorical_groups.append(group_vars)
        relaxable_categorical_inputs.extend(group_vars)
        new_constraints.append(pick_1_constraint)

    # create new domain with continuous inputs
    new_domain = Domain(
        inputs=kept_inputs + [var for key, (var, values) in relaxable_discrete_inputs.items()] + relaxable_categorical_inputs,  # type: ignore
        outputs=domain.outputs.features,  # type: ignore
        constraints=domain.constraints + new_constraints,
    )

    return new_domain, categorical_groups, relaxable_discrete_inputs  # type: ignore


def nchoosek_to_relaxable_domain_mapper(
    domain: Domain,
) -> Tuple[Domain, List[List[ContinuousInput]]]:
    var_occuring_in_nchoosek = []
    new_categories = []
    new_constraints = []
    n_choose_k_constraints = domain.constraints.get(includes=NChooseKConstraint)

    for constr in n_choose_k_constraints:
        var_occuring_in_nchoosek.extend(constr.features)  # type: ignore

        current_features: List[Feature] = [
            domain.get_feature(k) for k in constr.features  # type: ignore
        ]
        new_relaxable_categorical_vars, new_nchoosek_constraints = NChooseKGroup(
            current_features, constr.min_count, constr.max_count, constr.none_also_valid  # type: ignore
        )
        new_categories.append(new_relaxable_categorical_vars)
        new_constraints.extend(new_nchoosek_constraints)

        # allow vars to be set to 0
        for var in var_occuring_in_nchoosek:
            current_var = domain.inputs.get_by_key(var)
            if current_var.lower_bound > 0:  # type: ignore
                current_var.bounds = (0, current_var.upper_bound)  # type: ignore
            elif current_var.upper_bound < 0:  # type: ignore
                current_var.bounds = (current_var.lower_bound, 0)  # type: ignore

    new_domain = Domain(
        inputs=domain.inputs.features + [var for group in new_categories for var in group],  # type: ignore
        outputs=domain.outputs,
        constraints=domain.constraints.get(excludes=NChooseKConstraint)
        + new_constraints,
    )  # type: ignore
    return new_domain, new_categories


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
    Union[List[CategoricalInput], List[ContinuousInput]],
    List[ContinuousInput],
    List[
        Union[
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NonlinearInequalityConstraint,
        ]
    ],
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
            If False, ContinuousInput is used in combination with LinearConstraints as a relaxation.
            If True, CategoricalInput used in combination with NonlinearConstraints, as CategoricalInput can not be
            used within LinearConstraints
    Returns:
        Either one CategoricalInput wrapped in a List or List of ContinuousInput describing the group,
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
        quantity_if_picked = [quantity_if_picked for k in keys]  # type: ignore

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

    # choosing between CategoricalInput and ContinuousInput
    if use_non_relaxable_category_and_non_linear_constraint:
        category = [CategoricalInput(key=unique_group_identifier, categories=keys)]
        # if we use_legacy_class is true this constraint will be added by the discrete_to_relaxable_domain_mapper function
        pick_exactly_one_of_group_const = []
    else:
        category = [ContinuousInput(key=k, bounds=(0, 1)) for k in keys]
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
        all_new_constraints.append(max_quantity_constraint)  # type: ignore
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
) -> Tuple[
    List[ContinuousInput],
    Union[List[NonlinearInequalityConstraint], List[LinearInequalityConstraint]],
    Union[List[NonlinearInequalityConstraint], List[LinearInequalityConstraint]],
    Optional[Union[LinearEqualityConstraint, LinearInequalityConstraint]],
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
                coefficients=[-1.0] + [q[0] for i in range(len(combi))],
                rhs=0,
            )
            for combi, k, q in zip(all_quantity_features, keys, quantity_if_picked)
            if len(combi) >= 1
        ]

        quantity_constraints_ub = [
            LinearInequalityConstraint(
                features=[unique_group_identifier + "_" + k + "_quantity"] + combi,
                coefficients=[1.0] + [-q[1] for i in range(len(combi))],
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
    variables: List[Feature],
    pick_at_least: int,
    pick_at_most: int,
    none_also_valid: bool,
) -> Tuple[
    List[ContinuousInput],
    List[Union[LinearEqualityConstraint, LinearInequalityConstraint]],
]:
    """
    helper function to generate an N choose K problem with categorical variables, with an option to connect each
    element of a category to a corresponding quantity of how much that category should be used.

    Args:
        variables (List[ContinuousInput]): variables to pick from
        pick_at_least (int): minimum number of elements to be picked from the category. >=0
        pick_at_most (int): maximum number of elements to be picked from the category. >=pick_at_least
        none_also_valid (bool): defines if also none of the elements can be picked
    Returns:
        List of ContinuousInput describing the group,
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
            coefficients=[-1.0] + [var.lower_bound for i in range(len(combi))],  # type: ignore
            rhs=0,
        )
        for combi, var in zip(valid_states, variables)
        if len(combi) >= 1
    ]

    quantity_constraints_ub = [
        LinearInequalityConstraint(
            features=[var.key] + combi,
            coefficients=[1.0] + [-var.upper_bound for i in range(len(combi))],  # type: ignore
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

    category = [ContinuousInput(key=k, bounds=(0, 1)) for k in keys]
    pick_exactly_one_of_group_const = [
        LinearEqualityConstraint(
            features=list(keys), coefficients=[1 for k in keys], rhs=1
        )
    ]

    all_new_constraints = []
    all_new_constraints.extend(pick_exactly_one_of_group_const)
    all_new_constraints.extend(quantity_constraints_lb)
    all_new_constraints.extend(quantity_constraints_ub)

    return category, all_new_constraints


def generate_mixture_constraints(
    keys: List[str],
) -> Tuple[LinearEqualityConstraint, List[ContinuousInput]]:
    binary_vars = (ContinuousInput(key=x, bounds=(0, 1)) for x in keys)

    mixture_constraint = LinearEqualityConstraint(
        features=keys, coefficients=[1 for x in range(len(keys))], rhs=1
    )

    return mixture_constraint, list(binary_vars)


def design_from_original_to_new_domain(
    original_domain: Domain, new_domain: Domain, design: pd.DataFrame
) -> pd.DataFrame:
    raise NotImplementedError(
        "mapping a design to a new domain is not implemented yet."
    )


def design_from_new_to_original_domain(
    original_domain: Domain, design: pd.DataFrame
) -> pd.DataFrame:
    # map the ContinuousInput describing the categoricals to the corresponding CategoricalInputs, choose random for multiple solutions
    transformed_design = design[
        original_domain.get_feature_keys(excludes=[CategoricalInput, Output])
    ]

    for group in original_domain.get_features(includes=CategoricalInput):
        categorical_columns = design[group.categories]  # type: ignore
        mask = ~np.isclose(categorical_columns.to_numpy(), 0)

        for i, row in enumerate(mask):
            non_zero_indices = np.argwhere(row)
            if non_zero_indices.size != 0:
                index_to_keep = np.random.choice(np.argwhere(row).flatten())
            else:
                index_to_keep = list(range(len(row)))
            mask[i] = np.zeros_like(row, dtype=bool)
            mask[i][index_to_keep] = True

        categorical_columns = categorical_columns.where(
            np.invert(mask),
            pd.DataFrame(
                np.full(
                    (len(categorical_columns), len(group.categories)),  # type: ignore
                    group.categories,  # type: ignore
                ),
                columns=categorical_columns.columns,
                index=categorical_columns.index,
            ),
        )
        categorical_columns = categorical_columns.where(
            mask,
            pd.DataFrame(
                np.full((len(categorical_columns), len(group.categories)), ""),  # type: ignore
                columns=categorical_columns.columns,
                index=categorical_columns.index,
            ),
        )
        transformed_design[group.key] = categorical_columns.apply("".join, axis=1)

    # map the ContinuousInput describing the discrete to the closest valid value
    for var in original_domain.get_features(includes=DiscreteInput):
        closest_solution = var.from_continuous(transformed_design)  # type: ignore
        transformed_design[var.key] = closest_solution

    return transformed_design


def equal_count_split(
    values: List[float], lower_bound: float, upper_bound: float
) -> Tuple[float, float]:
    """
    Determines the two elements x and y such that the intervals (lower_bound, x) and (y, upper_bound)
    have the same number of elements regarding the values of the discrete variable
    Args:
        values: the values to split into a range
        lower_bound: inclusive lower bound
        upper_bound: inclusive upper bound

    Returns: tuple of floats which split the interval in half

    """
    values.sort()
    sub_list = [elem for elem in values if lower_bound <= elem <= upper_bound]

    size = len(sub_list)
    if size % 2 == 0:
        lower_index = size / 2 - 1
        upper_index = size / 2
    elif size == 1:
        return sub_list[0], sub_list[0]
    else:
        lower_index = math.floor(size / 2)
        upper_index = math.ceil(size / 2)

    lower_index = int(lower_index)
    upper_index = int(upper_index)

    return sub_list[lower_index], sub_list[upper_index]
