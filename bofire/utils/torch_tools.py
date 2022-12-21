from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from bofire.domain import Domain
from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.features import InputFeature

tkwargs = {
    "dtype": torch.double,
    "device": "cpu",
}


def get_linear_constraints(
    domain: Domain,
    constraint: Union[LinearEqualityConstraint, LinearInequalityConstraint],
    unit_scaled: bool = False,
) -> List[Tuple[Tensor, Tensor, float]]:
    """Converts linear constraints to the form required by BoTorch.

    Args:
        domain (Domain): Optimization problem definition.
        constraint (Union[LinearEqualityConstraint, LinearInequalityConstraint]): Type of constraint that should be converted.
        unit_scaled (bool, optional): If True, transforms constraints by assuming that the bound for the continuous features are [0,1]. Defaults to False.

    Returns:
        List[Tuple[Tensor, Tensor, float]]: List of tuples, each tuple consists of a tensor with the feature indices, coefficients and a float for the rhs.
    """
    constraints = []
    for c in domain.cnstrs.get(constraint):
        indices = []
        coefficients = []
        lower = []
        upper = []
        rhs = 0.0
        for i, featkey in enumerate(c.features):  # type: ignore
            idx = domain.get_feature_keys(InputFeature).index(featkey)
            feat = domain.get_feature(featkey)
            if feat.is_fixed():  # type: ignore
                rhs -= feat.fixed_value() * c.coefficients[i]  # type: ignore
            else:
                lower.append(feat.lower_bound)  # type: ignore
                upper.append(feat.upper_bound)  # type: ignore
                indices.append(idx)
                coefficients.append(
                    c.coefficients[i]  # type: ignore
                )  # if unit_scaled == False else c_scaled.coefficients[i])
        if unit_scaled:
            lower = np.array(lower)
            upper = np.array(upper)
            s = upper - lower
            scaled_coefficients = s * np.array(coefficients)
            constraints.append(
                (
                    torch.tensor(indices),
                    -torch.tensor(scaled_coefficients).to(**tkwargs),
                    -(rhs + c.rhs - np.sum(np.array(coefficients) * lower)),  # type: ignore
                )
            )
        else:
            constraints.append(
                (
                    torch.tensor(indices),
                    -torch.tensor(coefficients).to(**tkwargs),
                    -(rhs + c.rhs),  # type: ignore
                )
            )
    return constraints
