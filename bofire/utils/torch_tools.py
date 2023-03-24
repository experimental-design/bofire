from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from bofire.data_models.api import AnyObjective, Domain, Outputs
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.features.api import ContinuousInput, Input
from bofire.data_models.objectives.api import (
    BotorchConstrainedObjective,
    CloseToTargetObjective,
    DeltaObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    TargetObjective,
)

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
            idx = domain.get_feature_keys(Input).index(featkey)
            feat = domain.get_feature(featkey)
            if feat.is_fixed():  # type: ignore
                rhs -= feat.fixed_value()[0] * c.coefficients[i]  # type: ignore
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


def get_nchoosek_constraints(domain: Domain) -> List[Callable[[Tensor], float]]:
    """Transforms NChooseK constraints into a list of non-linear inequality constraint callables
    that can be parsed by pydantic. For this purpose the NChooseK constraint is continuously
    relaxed by countig the number of zeros in a candidate by a sum of narrow gaussians centered
    at zero.

    Args:
        domain (Domain): Optimization problem definition.

    Returns:
        List[Callable[[Tensor], float]]: List of callables that can be used
            as nonlinear equality constraints in botorch.
    """

    def narrow_gaussian(x, ell=1e-3):
        return torch.exp(-0.5 * (x / ell) ** 2)

    constraints = []
    # ignore none also valid for the start
    for c in domain.cnstrs.get(NChooseKConstraint):
        assert isinstance(c, NChooseKConstraint)
        indices = torch.tensor(
            [domain.get_feature_keys(ContinuousInput).index(key) for key in c.features],
            dtype=torch.int64,
        )
        if c.max_count != len(c.features):
            constraints.append(
                lambda x: narrow_gaussian(x=x[..., indices]).sum(dim=-1)
                - (len(c.features) - c.max_count)  # type: ignore
            )
        if c.min_count > 0:
            constraints.append(
                lambda x: -narrow_gaussian(x=x[..., indices]).sum(dim=-1)
                + (len(c.features) - c.min_count)  # type: ignore
            )
    return constraints


def get_output_constraints(
    output_features: Outputs,
) -> Tuple[List[Callable[[Tensor], Tensor]], List[float]]:
    """Method to translate output constraint objectives into a list of
    callables and list of etas for use in botorch.

    Args:
        output_features (Outputs): Output feature object that should
            be processed.

    Returns:
        Tuple[List[Callable[[Tensor], Tensor]], List[float]]: List of constraint callables,
            list of associated etas.
    """
    constraints = []
    etas = []
    for idx, feat in enumerate(output_features.get()):
        if isinstance(feat.objective, BotorchConstrainedObjective):  # type: ignore
            iconstraints, ietas = feat.objective.to_constraints(idx=idx)  # type: ignore
            constraints += iconstraints
            etas += ietas
    return constraints, etas


def get_objective_callable(
    idx: int, objective: AnyObjective
) -> Callable[[Tensor], Tensor]:  # type: ignore
    """Return a callable function for the given objective and index.

    The returned function is used to compute the objective value for a given set of inputs.
    The input index corresponds to the position of the variable for which the objective
    is being computed.

    Parameters:
        idx (int): The index of the input variable.
        objective (AnyObjective): An instance of the `AnyObjective` class.

    Returns:
        Callable[[Tensor], Tensor]: A callable function that takes in a tensor as input
        and returns the objective value as a tensor.

    Raises:
        NotImplementedError: If the objective type is not implemented.
    """

    if isinstance(objective, MaximizeObjective):
        return lambda x: (
            (x[..., idx] - objective.lower_bound)
            / (objective.upper_bound - objective.lower_bound)
        )
    if isinstance(objective, MinimizeObjective):
        return lambda x: -1.0 * (
            (x[..., idx] - objective.lower_bound)
            / (objective.upper_bound - objective.lower_bound)
        )
    if isinstance(objective, CloseToTargetObjective):
        return lambda x: -1.0 * (
            torch.abs(x[..., idx] - objective.target_value) ** objective.exponent
        )
    if isinstance(objective, MinimizeSigmoidObjective):
        return lambda x: (
            (
                1.0
                - 1.0
                / (
                    1.0
                    + torch.exp(
                        -1.0 * objective.steepness * (x[..., idx] - objective.tp)
                    )
                )
            )
        )
    if isinstance(objective, MaximizeSigmoidObjective):
        return lambda x: (
            1.0
            / (
                1.0
                + torch.exp(-1.0 * objective.steepness * ((x[..., idx] - objective.tp)))
            )
        )
    if isinstance(objective, TargetObjective):
        return lambda x: (
            1.0
            / (
                1.0
                + torch.exp(
                    -1
                    * objective.steepness
                    * (x[..., idx] - (objective.target_value - objective.tolerance))
                )
            )
            * (
                1.0
                - 1.0
                / (
                    1.0
                    + torch.exp(
                        -1.0
                        * objective.steepness
                        * (x[..., idx] - (objective.target_value + objective.tolerance))
                    )
                )
            )
        )
    if isinstance(objective, DeltaObjective):
        return lambda x: (objective.ref_point - x[..., idx]) * objective.scale
    else:
        raise NotImplementedError(
            f"Objective {objective.__class__.__name__} not implemented."
        )


def get_multiplicative_botorch_objective(
    output_features: Outputs,
) -> Callable[[Tensor, Tensor], Tensor]:
    """
    Returns a multiplicative botorch objective function that can be used for optimization.
    The returned function is a callable that takes two arguments:
    - samples (a `torch.Tensor` of shape `batch_size x q x m`): the samples at which to evaluate the objective.
    - X (a `torch.Tensor` of shape `n x d`): the input data for the underlying model.

    Parameters:
    - output_features (an instance of `Outputs`): the output features for which to construct the objective function.

    Returns:
    - a callable: the multiplicative botorch objective function. The function takes two arguments and returns a scalar `torch.Tensor`.
      The first argument is a `torch.Tensor` of shape `batch_size x q x m`, where `batch_size` is the number of independent batches to evaluate,
      `q` is the number of fantasy points to generate, and `m` is the dimensionality of the outcome space.
      The second argument is a `torch.Tensor` of shape `n x d`, where `n` is the number of training examples and `d` is the dimensionality of the input space.
    """

    callables = [
        get_objective_callable(idx=i, objective=feat.objective)  # type: ignore
        for i, feat in enumerate(output_features.get())
        if feat.objective is not None  # type: ignore
    ]
    weights = [
        feat.objective.w  # type: ignore
        for i, feat in enumerate(output_features.get())
        if feat.objective is not None  # type: ignore
    ]

    def objective(samples: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        val = 1.0
        for c, w in zip(callables, weights):
            val *= c(samples) ** w
        return val  # type: ignore

    return objective


def get_additive_botorch_objective(
    output_features: Outputs, exclude_constraints: bool = True
) -> Callable[[Tensor, Tensor], Tensor]:
    callables = [
        get_objective_callable(idx=i, objective=feat.objective)  # type: ignore
        for i, feat in enumerate(output_features.get())
        if feat.objective is not None  # type: ignore
        and (
            not isinstance(feat.objective, BotorchConstrainedObjective)  # type: ignore
            if exclude_constraints
            else True
        )
    ]
    weights = [
        feat.objective.w  # type: ignore
        for i, feat in enumerate(output_features.get())
        if feat.objective is not None  # type: ignore
        and (
            not isinstance(feat.objective, BotorchConstrainedObjective)  # type: ignore
            if exclude_constraints
            else True
        )
    ]

    def objective(samples: Tensor, X: Tensor) -> Tensor:
        val = 0.0
        for c, w in zip(callables, weights):
            val += c(samples) * w
        return val  # type: ignore

    return objective
