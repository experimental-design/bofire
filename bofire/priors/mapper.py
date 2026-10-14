import math
from typing import Callable, Optional, Type

import gpytorch
from botorch.utils.constraints import LogTransformedInterval, NonTransformedInterval
from gpytorch.constraints import GreaterThan, LessThan, Positive

import bofire.data_models.priors.api as data_models


def register(
    data_model_cls: Type,
    map_fn: Optional[Callable] = None,
):
    """Register a custom prior or constraint mapping from data model to factory function.

    Constraints (subclasses of `PriorConstraint` or `Interval`) are mapped by
    `map_constraint`, priors by `map`.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyPriorDataModel)
        def map_my_prior(data_model, **kwargs):
            return MyGpytorchPrior(...)

        # Direct call form
        register(MyPriorDataModel, map_my_prior)

    Args:
        data_model_cls: The Pydantic data model class.
        map_fn: A callable that takes ``(data_model, **kwargs)`` and returns a
            gpytorch prior or constraint. If not provided, returns a decorator.

    Returns:
        The mapping function (unchanged) when used as a decorator, None otherwise.
    """

    def _register(fn: Callable) -> Callable:
        # Register with the data model unions first so a discriminator conflict
        # is raised before the functional map is touched (no partial state).
        if issubclass(data_model_cls, data_models.Prior):
            data_models.register_prior(data_model_cls)
        elif issubclass(
            data_model_cls, (data_models.PriorConstraint, data_models.Interval)
        ):
            data_models.register_prior_constraint(data_model_cls)

        if issubclass(
            data_model_cls, (data_models.PriorConstraint, data_models.Interval)
        ):
            CONSTRAINT_MAP[data_model_cls] = fn
        else:
            PRIOR_MAP[data_model_cls] = fn

        return fn

    if map_fn is not None:
        _register(map_fn)
        return None

    return _register


def map_NormalPrior(
    data_model: data_models.NormalPrior,
    **kwargs,
) -> gpytorch.priors.NormalPrior:
    return gpytorch.priors.NormalPrior(loc=data_model.loc, scale=data_model.scale)


def map_GammaPrior(
    data_model: data_models.GammaPrior,
    **kwargs,
) -> gpytorch.priors.GammaPrior:
    return gpytorch.priors.GammaPrior(
        concentration=data_model.concentration,
        rate=data_model.rate,
    )


def map_LKJPrior(
    data_model: data_models.LKJPrior,
    **kwargs,
) -> gpytorch.priors.LKJPrior:
    return gpytorch.priors.LKJCovariancePrior(
        n=data_model.n_tasks,
        eta=data_model.shape,
        sd_prior=map(data_model.sd_prior),
    )


def map_LogNormalPrior(
    data_model: data_models.LogNormalPrior,
    **kwargs,
) -> gpytorch.priors.LogNormalPrior:
    return gpytorch.priors.LogNormalPrior(loc=data_model.loc, scale=data_model.scale)


def map_DimensionalityScaledLogNormalPrior(
    data_model: data_models.DimensionalityScaledLogNormalPrior,
    d: int,
) -> gpytorch.priors.LogNormalPrior:
    # a dimensionality-scaled log-normal is a plain log-normal with effective params
    return map_LogNormalPrior(
        data_models.LogNormalPrior(
            loc=data_model.loc + math.log(d) * data_model.loc_scaling,
            scale=(data_model.scale**2 + math.log(d) * data_model.scale_scaling) ** 0.5,
        )
    )


def map_DimensionalityScaledGammaPrior(
    data_model: data_models.DimensionalityScaledGammaPrior,
    d: int,
) -> gpytorch.priors.GammaPrior:
    # a dimensionality-scaled gamma is a plain gamma with effective params
    return map_GammaPrior(
        data_models.GammaPrior(
            concentration=data_model.concentration
            + math.sqrt(d) * data_model.concentration_scaling,
            rate=data_model.rate * d**data_model.rate_power,
        )
    )


def map_SmoothedBoxPrior(
    data_model: data_models.SmoothedBoxPrior,
    **kwargs,
) -> gpytorch.priors.smoothed_box_prior.SmoothedBoxPrior:
    return gpytorch.priors.smoothed_box_prior.SmoothedBoxPrior(
        a=data_model.lower_bound, b=data_model.upper_bound, sigma=data_model.sigma
    )


def map_Interval(
    data_model: data_models.Interval,
) -> gpytorch.constraints.Interval:
    return gpytorch.constraints.Interval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=data_model.initial_value,
    )


def map_NonTransformedInterval(
    data_model: data_models.NonTransformedInterval,
) -> NonTransformedInterval:
    return NonTransformedInterval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=data_model.initial_value,
    )


def map_LogTransformedInterval(
    data_model: data_models.LogTransformedInterval,
) -> LogTransformedInterval:
    return LogTransformedInterval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=data_model.initial_value,
    )


def map_Positive(
    data_model: data_models.Positive,
) -> Positive:
    return Positive(initial_value=data_model.initial_value)


def map_GreaterThan(
    data_model: data_models.GreaterThan,
) -> GreaterThan:
    return GreaterThan(
        lower_bound=data_model.lower_bound,
        transform=None,
        initial_value=data_model.initial_value,
    )


def map_LessThan(
    data_model: data_models.LessThan,
) -> LessThan:
    return LessThan(
        upper_bound=data_model.upper_bound,
        transform=None,
        initial_value=data_model.initial_value,
    )


PRIOR_MAP = {
    data_models.NormalPrior: map_NormalPrior,
    data_models.GammaPrior: map_GammaPrior,
    data_models.LKJPrior: map_LKJPrior,
    data_models.LogNormalPrior: map_LogNormalPrior,
    data_models.DimensionalityScaledLogNormalPrior: map_DimensionalityScaledLogNormalPrior,
    data_models.DimensionalityScaledGammaPrior: map_DimensionalityScaledGammaPrior,
    data_models.SmoothedBoxPrior: map_SmoothedBoxPrior,
}

CONSTRAINT_MAP = {
    data_models.Interval: map_Interval,
    data_models.NonTransformedInterval: map_NonTransformedInterval,
    data_models.LogTransformedInterval: map_LogTransformedInterval,
    data_models.Positive: map_Positive,
    data_models.GreaterThan: map_GreaterThan,
    data_models.LessThan: map_LessThan,
}


def map(data_model: data_models.AnyPrior, **kwargs) -> gpytorch.priors.Prior:
    """Build the GPyTorch prior for a prior data model.

    Args:
        data_model: The prior.
        **kwargs: Passed to the prior's mapper, e.g. `d`, the number of input
            dimensions, for dimensionality-scaled priors.

    Returns:
        The GPyTorch prior.
    """
    return PRIOR_MAP[data_model.__class__](data_model, **kwargs)


def map_constraint(
    data_model: Optional[data_models.AnyPriorConstraint],
    prior: Optional[gpytorch.priors.Prior] = None,
    default: Callable[..., gpytorch.constraints.Interval] = Positive,
) -> Optional[gpytorch.constraints.Interval]:
    """The constraint of a hyperparameter, starting its fit where it should start.

    The fit starts at the constraint's `initial_value` if it gives one. Otherwise it
    starts at the mode of the hyperparameter's prior, moved into the bounds, as
    BoTorch's default models do; for a transformed bound only if the mode lies inside
    it. Without a usable mode, a bound without a transform starts inside it: at the
    lower bound plus ln 2, the upper bound minus ln 2, or the middle of an interval.

    Args:
        data_model: The constraint, if any.
        prior: The GPyTorch prior of the same hyperparameter, if any.
        default: GPyTorch's own constraint for the hyperparameter, used without a
            constraint so that the fit can start at the prior's mode.

    Returns:
        The constraint, or `None` to keep GPyTorch's own constraint and start.
    """
    mode = _prior_mode(prior)
    if data_model is None:
        if mode is None or _start(default(), mode) is None:
            return None
        return default(initial_value=mode)
    to_constraint = CONSTRAINT_MAP[data_model.__class__]
    constraint = to_constraint(data_model)
    if data_model.initial_value is not None:
        return constraint
    start = _start(constraint, mode)
    if start is None:
        return constraint
    return to_constraint(data_model.model_copy(update={"initial_value": start}))


def _prior_mode(prior: Optional[gpytorch.priors.Prior]) -> Optional[float]:
    """The mode of a prior, if it has a single one."""
    if prior is None:
        return None
    try:
        mode = prior.mode
    except (NotImplementedError, AttributeError):
        return None
    if mode.numel() != 1:
        return None
    return float(mode)


def _start(
    constraint: gpytorch.constraints.Interval, mode: Optional[float]
) -> Optional[float]:
    """Where the fit starts under a constraint; `None` keeps GPyTorch's start."""
    lower = float(constraint.lower_bound)
    upper = float(constraint.upper_bound)
    if mode is not None:
        if not constraint.enforced:
            return min(max(mode, lower), upper)
        if lower < mode < upper:
            return mode
    if constraint.enforced:
        return None
    if math.isfinite(lower) and math.isfinite(upper):
        return (lower + upper) / 2
    if math.isfinite(lower):
        return lower + math.log(2)
    if math.isfinite(upper):
        return upper - math.log(2)
    return None
