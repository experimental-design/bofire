import math
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from bofire.data_models.api import AnyObjective, Domain, Outputs
from bofire.data_models.constraints.api import (
    InterpointEqualityConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.features.api import ContinuousInput, Input
from bofire.data_models.objectives.api import (
    CloseToTargetObjective,
    ConstrainedCategoricalObjective,
    ConstrainedObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
    MinimizeSigmoidObjective,
    MovingMaximizeSigmoidObjective,
    TargetObjective,
)
from bofire.strategies.strategy import Strategy


tkwargs = {
    "dtype": torch.double,
    "device": "cpu",
}


def get_linear_constraints(
    domain: Domain,
    constraint: Union[Type[LinearEqualityConstraint], Type[LinearInequalityConstraint]],
    unit_scaled: bool = False,
) -> List[Tuple[Tensor, Tensor, float]]:
    """Converts linear constraints to the form required by BoTorch.

    Args:
        domain: Optimization problem definition.
        constraint: Type of constraint that should be converted.
        unit_scaled: If True, transforms constraints by assuming that the bound for the continuous features are [0,1]. Defaults to False.

    Returns:
        List[Tuple[Tensor, Tensor, float]]: List of tuples, each tuple consists of a tensor with the feature indices, coefficients and a float for the rhs.
    """
    constraints = []
    for c in domain.constraints.get(constraint):
        indices = []
        coefficients = []
        lower = []
        upper = []
        rhs = 0.0
        for i, featkey in enumerate(c.features):
            idx = domain.inputs.get_keys(Input).index(featkey)
            feat = domain.inputs.get_by_key(featkey)
            if feat.is_fixed():
                rhs -= feat.fixed_value()[0] * c.coefficients[i]  # type: ignore
            else:
                lower.append(feat.lower_bound)  # type: ignore
                upper.append(feat.upper_bound)  # type: ignore
                indices.append(idx)
                coefficients.append(
                    c.coefficients[i]
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
                    -(rhs + c.rhs - np.sum(np.array(coefficients) * lower)),
                )
            )
        else:
            constraints.append(
                (
                    torch.tensor(indices),
                    -torch.tensor(coefficients).to(**tkwargs),
                    -(rhs + c.rhs),
                )
            )
    return constraints


def get_interpoint_constraints(
    domain: Domain, n_candidates: int
) -> List[Tuple[Tensor, Tensor, float]]:
    """Converts interpoint equality constraints to linear equality constraints,
        that can be processed by botorch. For more information, see the docstring
        of `optimize_acqf` in botorch
        (https://github.com/pytorch/botorch/blob/main/botorch/optim/optimize.py).

    Args:
        domain (Domain): Optimization problem definition.
        n_candidates (int): Number of candidates that should be requested.

    Returns:
        List[Tuple[Tensor, Tensor, float]]: List of tuples, each tuple consists
            of a tensor with the feature indices, coefficients and a float for the rhs.
    """
    constraints = []
    if n_candidates == 1:
        return constraints
    for constraint in domain.constraints.get(InterpointEqualityConstraint):
        assert isinstance(constraint, InterpointEqualityConstraint)
        coefficients = torch.tensor([1.0, -1.0]).to(**tkwargs)
        feat_idx = domain.inputs.get_keys(Input).index(constraint.feature)
        feat = domain.inputs.get_by_key(constraint.feature)
        assert isinstance(feat, ContinuousInput)
        if feat.is_fixed():
            continue
        multiplicity = constraint.multiplicity or n_candidates
        for i in range(math.ceil(n_candidates / multiplicity)):
            all_indices = torch.arange(
                i * multiplicity, min((i + 1) * multiplicity, n_candidates)
            )
            for k in range(len(all_indices) - 1):
                indices = torch.tensor(
                    [[all_indices[0], feat_idx], [all_indices[k + 1], feat_idx]],
                    dtype=torch.int64,
                )
                constraints.append((indices, coefficients, 0.0))
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

    def max_constraint(indices: Tensor, num_features: int, max_count: int):
        return lambda x: narrow_gaussian(x=x[..., indices]).sum(dim=-1) - (
            num_features - max_count
        )

    def min_constraint(indices: Tensor, num_features: int, min_count: int):
        return lambda x: -narrow_gaussian(x=x[..., indices]).sum(dim=-1) + (
            num_features - min_count
        )

    constraints = []
    # ignore none also valid for the start
    for c in domain.constraints.get(NChooseKConstraint):
        assert isinstance(c, NChooseKConstraint)
        indices = torch.tensor(
            [domain.inputs.get_keys(ContinuousInput).index(key) for key in c.features],
            dtype=torch.int64,
        )
        if c.max_count != len(c.features):
            constraints.append(
                max_constraint(
                    indices=indices, num_features=len(c.features), max_count=c.max_count
                )
            )
        if c.min_count > 0:
            constraints.append(
                min_constraint(
                    indices=indices, num_features=len(c.features), min_count=c.min_count
                )
            )
    return constraints


def get_product_constraints(domain: Domain) -> List[Callable[[Tensor], float]]:
    """
    Returns a list of nonlinear constraint functions that can be processed by botorch
    based on the given domain.

    Args:
        domain (Domain): The domain object containing the constraints.

    Returns:
        List[Callable[[Tensor], float]]: A list of product constraint functions.

    """

    def product_constraint(indices: Tensor, exponents: Tensor, rhs: float, sign: int):
        return lambda x: -1.0 * sign * (x[..., indices] ** exponents).prod(dim=-1) + rhs

    constraints = []
    for c in domain.constraints.get(ProductInequalityConstraint):
        assert isinstance(c, ProductInequalityConstraint)
        indices = torch.tensor(
            [domain.inputs.get_keys(ContinuousInput).index(key) for key in c.features],
            dtype=torch.int64,
        )
        constraints.append(
            product_constraint(indices, torch.tensor(c.exponents), c.rhs, c.sign)
        )
    return constraints


def get_nonlinear_constraints(domain: Domain) -> List[Callable[[Tensor], float]]:
    """
    Returns a list of callable functions that represent the nonlinear constraints
    for the given domain that can be processed by botorch.

    Parameters:
        domain (Domain): The domain for which to generate the nonlinear constraints.

    Returns:
        List[Callable[[Tensor], float]]: A list of callable functions that take a tensor
        as input and return a float value representing the constraint evaluation.
    """
    return get_nchoosek_constraints(domain) + get_product_constraints(domain)


def constrained_objective2botorch(
    idx: int,
    objective: ConstrainedObjective,
    x_adapt: Optional[Tensor],
    eps: float = 1e-8,
) -> Tuple[List[Callable[[Tensor], Tensor]], List[float], int]:
    """Create a callable that can be used by `botorch.utils.objective.apply_constraints`
    to setup ouput constrained optimizations.

    Args:
        idx (int): Index of the constraint objective in the list of outputs.
        objective (BotorchConstrainedObjective): The objective that should be transformed.
        x_adapt (Optional[Tensor]): The tensor that should be used to adapt the objective,
            for example in case of a moving turning point in the `MovingMaximizeSigmoidObjective`.
        eps (float, optional): Small value to avoid numerical instabilities in case of the `ConstrainedCategoricalObjective`.
            Defaults to 1e-8.

    Returns:
        Tuple[List[Callable[[Tensor], Tensor]], List[float], int]: List of callables that can be used by botorch for setting up the constrained objective,
            list of the corresponding botorch eta values, final index used by the method (to track for categorical variables)
    """
    assert isinstance(
        objective, ConstrainedObjective
    ), "Objective is not a `ConstrainedObjective`."
    if isinstance(objective, MaximizeSigmoidObjective):
        return (
            [lambda Z: (Z[..., idx] - objective.tp) * -1.0],
            [1.0 / objective.steepness],
            idx + 1,
        )
    elif isinstance(objective, MovingMaximizeSigmoidObjective):
        assert x_adapt is not None
        tp = x_adapt.max().item() + objective.tp
        return (
            [lambda Z: (Z[..., idx] - tp) * -1.0],
            [1.0 / objective.steepness],
            idx + 1,
        )
    elif isinstance(objective, MinimizeSigmoidObjective):
        return (
            [lambda Z: (Z[..., idx] - objective.tp)],
            [1.0 / objective.steepness],
            idx + 1,
        )
    elif isinstance(objective, TargetObjective):
        return (
            [
                lambda Z: (Z[..., idx] - (objective.target_value - objective.tolerance))
                * -1.0,
                lambda Z: (
                    Z[..., idx] - (objective.target_value + objective.tolerance)
                ),
            ],
            [1.0 / objective.steepness, 1.0 / objective.steepness],
            idx + 1,
        )
    elif isinstance(objective, ConstrainedCategoricalObjective):
        # The output of a categorical objective has final dim `c` where `c` is number of classes
        # Pass in the expected acceptance probability and perform an inverse sigmoid to atain the original probabilities
        return (
            [
                lambda Z: torch.log(
                    1
                    / torch.clamp(
                        (
                            Z[..., idx : idx + len(objective.desirability)]
                            * torch.tensor(objective.desirability).to(**tkwargs)
                        ).sum(-1),
                        min=eps,
                        max=1 - eps,
                    )
                    - 1,
                )
            ],
            [1.0],
            idx + len(objective.desirability),
        )
    else:
        raise ValueError(f"Objective {objective.__class__.__name__} not known.")


def get_output_constraints(
    outputs: Outputs, experiments: pd.DataFrame
) -> Tuple[List[Callable[[Tensor], Tensor]], List[float]]:
    """Method to translate output constraint objectives into a list of
    callables and list of etas for use in botorch.

    Args:
        outputs (Outputs): Output feature object that should
            be processed.
        experiments (pd.DataFrame): DataFrame containing the experiments that are used for
            adapting the objectives on the fly, for example in the case of the
            `MovingMaximizeSigmoidObjective`.

    Returns:
        Tuple[List[Callable[[Tensor], Tensor]], List[float]]: List of constraint callables,
            list of associated etas.
    """
    constraints = []
    etas = []
    idx = 0
    for feat in outputs.get():
        if isinstance(feat.objective, ConstrainedObjective):
            cleaned_experiments = outputs.preprocess_experiments_one_valid_output(
                feat.key, experiments
            )
            iconstraints, ietas, idx = constrained_objective2botorch(
                idx,
                objective=feat.objective,
                x_adapt=torch.from_numpy(cleaned_experiments[feat.key].values).to(
                    **tkwargs
                )
                if not isinstance(feat.objective, ConstrainedCategoricalObjective)
                else None,
            )
            constraints += iconstraints
            etas += ietas
        else:
            idx += 1
    return constraints, etas


def get_objective_callable(
    idx: int, objective: AnyObjective, x_adapt: Tensor
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    if isinstance(objective, MaximizeObjective):
        return lambda y, X=None: (
            (y[..., idx] - objective.lower_bound)
            / (objective.upper_bound - objective.lower_bound)
        )
    if isinstance(objective, MinimizeObjective):
        return lambda y, X=None: -1.0 * (
            (y[..., idx] - objective.lower_bound)
            / (objective.upper_bound - objective.lower_bound)
        )
    if isinstance(objective, CloseToTargetObjective):
        return lambda y, X=None: -1.0 * (
            torch.abs(y[..., idx] - objective.target_value) ** objective.exponent
        )
    if isinstance(objective, MinimizeSigmoidObjective):
        return lambda y, X=None: (
            1.0
            - 1.0
            / (
                1.0
                + torch.exp(-1.0 * objective.steepness * (y[..., idx] - objective.tp))
            )
        )
    if isinstance(objective, MaximizeSigmoidObjective):
        return lambda y, X=None: (
            1.0
            / (
                1.0
                + torch.exp(-1.0 * objective.steepness * (y[..., idx] - objective.tp))
            )
        )
    if isinstance(objective, MovingMaximizeSigmoidObjective):
        tp = x_adapt.max().item() + objective.tp
        return lambda y, X=None: (
            1.0 / (1.0 + torch.exp(-1.0 * objective.steepness * (y[..., idx] - tp)))
        )
    if isinstance(objective, TargetObjective):
        return lambda y, X=None: (
            1.0
            / (
                1.0
                + torch.exp(
                    -1
                    * objective.steepness
                    * (y[..., idx] - (objective.target_value - objective.tolerance))
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
                        * (y[..., idx] - (objective.target_value + objective.tolerance))
                    )
                )
            )
        )
    else:
        raise NotImplementedError(
            f"Objective {objective.__class__.__name__} not implemented."
        )


def get_custom_botorch_objective(
    outputs: Outputs,
    f: Callable[
        [
            Tensor,
            List[Callable[[Tensor, Optional[Tensor]], Tensor]],
            List[float],
            Tensor,
        ],
        Tensor,
    ],
    experiments: pd.DataFrame,
    exclude_constraints: bool = True,
) -> Callable[[Tensor, Tensor], Tensor]:
    callables = [
        get_objective_callable(
            idx=i,
            objective=feat.objective,
            x_adapt=torch.from_numpy(
                outputs.preprocess_experiments_one_valid_output(feat.key, experiments)[
                    feat.key
                ].values
            ).to(**tkwargs),
        )
        for i, feat in enumerate(outputs.get())
        if feat.objective is not None
        and (
            not isinstance(feat.objective, ConstrainedObjective)
            if exclude_constraints
            else True
        )
    ]
    weights = [
        feat.objective.w
        for i, feat in enumerate(outputs.get())
        if feat.objective is not None
        and (
            not isinstance(feat.objective, ConstrainedObjective)
            if exclude_constraints
            else True
        )
    ]

    def objective(samples: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return f(samples, callables, weights, X)

    return objective


def get_multiplicative_botorch_objective(
    outputs: Outputs,
    experiments: pd.DataFrame,
) -> Callable[[Tensor, Tensor], Tensor]:
    callables = [
        get_objective_callable(
            idx=i,
            objective=feat.objective,
            x_adapt=torch.from_numpy(
                outputs.preprocess_experiments_one_valid_output(feat.key, experiments)[
                    feat.key
                ].values
            ).to(**tkwargs),
        )
        for i, feat in enumerate(outputs.get())
        if feat.objective is not None
    ]
    weights = [
        feat.objective.w
        for i, feat in enumerate(outputs.get())
        if feat.objective is not None
    ]

    def objective(samples: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        val = 1.0
        for c, w in zip(callables, weights):
            val *= c(samples, None) ** w
        return val  # type: ignore

    return objective


def get_additive_botorch_objective(
    outputs: Outputs,
    experiments: pd.DataFrame,
    exclude_constraints: bool = True,
) -> Callable[[Tensor, Tensor], Tensor]:
    callables = [
        get_objective_callable(
            idx=i,
            objective=feat.objective,
            x_adapt=torch.from_numpy(
                outputs.preprocess_experiments_one_valid_output(feat.key, experiments)[
                    feat.key
                ].values
            ).to(**tkwargs),
        )
        for i, feat in enumerate(outputs.get())
        if feat.objective is not None
        and (
            not isinstance(feat.objective, ConstrainedObjective)
            if exclude_constraints
            else True
        )
    ]
    weights = [
        feat.objective.w
        for i, feat in enumerate(outputs.get())
        if feat.objective is not None
        and (
            not isinstance(feat.objective, ConstrainedObjective)
            if exclude_constraints
            else True
        )
    ]

    def objective(samples: Tensor, X: Tensor) -> Tensor:
        val = 0.0
        for c, w in zip(callables, weights):
            val += c(samples, None) * w
        return val  # type: ignore

    return objective


def get_multiobjective_objective(
    outputs: Outputs, experiments: pd.DataFrame
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    """Returns a callable that can be used by botorch for multiobjective optimization.

    Args:
        outputs (Outputs): Outputs object for which the callable should be generated.
        experiments (pd.DataFrame): DataFrame containing the experiments that are used for
            adapting the objectives on the fly, for example in the case of the
            `MovingMaximizeSigmoidObjective`.

    Returns:
        Callable[[Tensor], Tensor]: _description_
    """
    callables = [
        get_objective_callable(
            idx=i,
            objective=feat.objective,
            x_adapt=torch.from_numpy(
                outputs.preprocess_experiments_one_valid_output(feat.key, experiments)[
                    feat.key
                ].values
            ).to(**tkwargs),
        )
        for i, feat in enumerate(outputs.get())
        if feat.objective is not None
        and isinstance(
            feat.objective,
            (MaximizeObjective, MinimizeObjective, CloseToTargetObjective),
        )
    ]

    def objective(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        return torch.stack([c(samples, None) for c in callables], dim=-1)

    return objective


def get_initial_conditions_generator(
    strategy: Strategy,
    transform_specs: Dict,
    ask_options: Optional[Dict] = None,
    sequential: bool = True,
) -> Callable[[int, int, int], Tensor]:
    """Takes a strategy object and returns a callable which uses this
    strategy to return a generator callable which can be used in botorch`s
    `gen_batch_initial_conditions` to generate samples.

    Args:
        strategy (Strategy): Strategy that should be used to generate samples.
        transform_specs (Dict): Dictionary indicating how the samples should be
            transformed.
        ask_options (Dict, optional): Dictionary of keyword arguments that are
            passed to the `ask` method of the strategy. Defaults to {}.
        sequential (bool, optional): If True, samples for every q-batch are
            generate indepenent from each other. If False, the `n x q` samples
            are generated at once.

    Returns:
        Callable[[int, int, int], Tensor]: Callable that can be passed to
            `batch_initial_conditions`.
    """
    if ask_options is None:
        ask_options = {}

    def generator(n: int, q: int, seed: int) -> Tensor:
        if sequential:
            initial_conditions = []
            for _ in range(n):
                candidates = strategy.ask(q, **ask_options)
                # transform it
                transformed_candidates = strategy.domain.inputs.transform(
                    candidates, transform_specs
                )
                # transform to tensor
                initial_conditions.append(
                    torch.from_numpy(transformed_candidates.values).to(**tkwargs)
                )
            return torch.stack(initial_conditions, dim=0)
        else:
            candidates = strategy.ask(n * q, **ask_options)
            # transform it
            transformed_candidates = strategy.domain.inputs.transform(
                candidates, transform_specs
            )
            return (
                torch.from_numpy(transformed_candidates.values)
                .to(**tkwargs)
                .reshape(n, q, transformed_candidates.shape[1])
            )

    return generator
