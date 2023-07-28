from itertools import product
from typing import List, Tuple

import numpy as np

from bofire.data_models.constraints.api import (
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import ContinuousBinaryInput


def constraints_mapper(
    constraints: List[List[LinearConstraint]], variables: Inputs
) -> List[LinearConstraint]:
    num_of_binary_vars = int(np.ceil(np.log2(len(constraints))))
    possible_fixations = list(product([False, True], repeat=num_of_binary_vars))
    np.power(2, num_of_binary_vars) - len(constraints)
    allowed_fixations = possible_fixations[: len(constraints)]
    possible_fixations[len(constraints) :]

    # todo ensure unique name for each var
    binary_vars = [
        ContinuousBinaryInput(key=f"a{i}") for i in range(num_of_binary_vars)
    ]
    binary_keys = [f"a{i}" for i in range(num_of_binary_vars)]

    # todo find tighter bounds
    transformed_constraints = []
    for k, group in enumerate(constraints):
        for const in group:
            max_bound = max((x.upper_bound for x in variables))
            new_keys = const.features + binary_keys
            coefficient_value = max_bound - const.rhs
            sign = allowed_fixations[k]
            binary_coefficients = [
                coefficient_value if sign[i] else -coefficient_value
                for i in range(len(binary_keys))
            ]
            new_coefficients = const.coefficients + binary_coefficients
            new_rhs = const.rhs + sum((1 if x else 0 for x in sign)) * coefficient_value

            if isinstance(const, LinearInequalityConstraint):
                new_const = LinearInequalityConstraint(
                    features=new_keys, coefficients=new_coefficients, rhs=new_rhs
                )
            elif isinstance(const, LinearEqualityConstraint):
                new_const = LinearEqualityConstraint(
                    features=new_keys, coefficients=new_coefficients, rhs=new_rhs
                )
            else:
                pass

            transformed_constraints.append(new_const)

    return transformed_constraints, binary_vars


def get_bounds_of_constraint(constraint: LinearConstraint, variables: Inputs):
    features = constraint.features
    coefficients = constraint.coefficients
    (True if x >= 0 else False for x in coefficients)

    for feat in features:
        variables.get_by_keys(feat)


def generate_mixture_constraints(
    keys: List[str], rhs: float = 1, equality=True
) -> Tuple[LinearEqualityConstraint, List[ContinuousBinaryInput]]:
    binary_vars = (ContinuousBinaryInput(key=x) for x in keys)

    if equality:
        mixture_constraint = LinearEqualityConstraint(
            features=keys, coefficients=[1 for x in range(len(keys))], rhs=rhs
        )
    else:
        mixture_constraint = LinearInequalityConstraint(
            features=keys, coefficients=[1 for x in range(len(keys))], rhs=rhs
        )

    return mixture_constraint, list(binary_vars)
