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
    data_model: data_models.Interval, prior: Optional[gpytorch.priors.Prior] = None
) -> gpytorch.constraints.Interval:
    return gpytorch.constraints.Interval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=_initial_value(
            data_model, prior, data_model.lower_bound, data_model.upper_bound
        ),
    )


def map_NonTransformedInterval(
    data_model: data_models.NonTransformedInterval,
    prior: Optional[gpytorch.priors.Prior] = None,
) -> NonTransformedInterval:
    return NonTransformedInterval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=_initial_value(
            data_model,
            prior,
            data_model.lower_bound,
            data_model.upper_bound,
            default=(data_model.lower_bound + data_model.upper_bound) / 2,
        ),
    )


def map_LogTransformedInterval(
    data_model: data_models.LogTransformedInterval,
    prior: Optional[gpytorch.priors.Prior] = None,
) -> LogTransformedInterval:
    return LogTransformedInterval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=_initial_value(
            data_model, prior, data_model.lower_bound, data_model.upper_bound
        ),
    )


def map_Positive(
    data_model: data_models.Positive, prior: Optional[gpytorch.priors.Prior] = None
) -> Positive:
    return Positive(initial_value=_initial_value(data_model, prior, lower=0.0))


def map_GreaterThan(
    data_model: data_models.GreaterThan, prior: Optional[gpytorch.priors.Prior] = None
) -> GreaterThan:
    return GreaterThan(
        lower_bound=data_model.lower_bound,
        transform=None,
        initial_value=_initial_value(
            data_model,
            prior,
            lower=data_model.lower_bound,
            default=data_model.lower_bound + math.log(2),
        ),
    )


def map_LessThan(
    data_model: data_models.LessThan, prior: Optional[gpytorch.priors.Prior] = None
) -> LessThan:
    return LessThan(
        upper_bound=data_model.upper_bound,
        transform=None,
        initial_value=_initial_value(
            data_model,
            prior,
            upper=data_model.upper_bound,
            default=data_model.upper_bound - math.log(2),
        ),
    )


def _initial_value(
    data_model: data_models.PriorConstraint,
    prior: Optional[gpytorch.priors.Prior],
    lower: float = -math.inf,
    upper: float = math.inf,
    default: Optional[float] = None,
) -> Optional[float]:
    """Where the fit starts: the constraint's own starting value, else the mode of the
    prior moved into the bounds, else `default` (`None` keeps GPyTorch's start)."""
    if data_model.initial_value is not None:
        return data_model.initial_value
    if prior is None:
        return default
    return min(max(mode(prior), lower), upper)


def mode(prior: gpytorch.priors.Prior) -> float:
    """The most likely value under a prior.

    A smoothed box is flat between its bounds, so it has no single mode; its middle
    is used.
    """
    if isinstance(prior, gpytorch.priors.SmoothedBoxPrior):
        return float((prior.a + prior.b) / 2)
    return float(prior.mode)


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
    data_model: data_models.AnyPriorConstraint,
    prior: Optional[gpytorch.priors.Prior] = None,
) -> gpytorch.constraints.Interval:
    """Build the GPyTorch constraint of a hyperparameter, with where its fit starts.

    The fit starts at the constraint's `initial_value` if it gives one, else at the
    mode of the hyperparameter's prior moved into the bounds, as BoTorch's default
    models do. Without either, a bound without a transform starts inside it: at the
    lower bound plus ln 2, the upper bound minus ln 2, or the middle of an interval.

    Args:
        data_model: The constraint.
        prior: The GPyTorch prior of the same hyperparameter, if it has one.

    Returns:
        The GPyTorch constraint.
    """
    return CONSTRAINT_MAP[data_model.__class__](data_model, prior=prior)
