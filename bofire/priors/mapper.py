import math
from typing import Callable, Optional, Type, Union

import gpytorch
from botorch.utils.constraints import LogTransformedInterval, NonTransformedInterval
from gpytorch.constraints import GreaterThan, LessThan, Positive

import bofire.data_models.priors.api as data_models


Constraint = Union[
    Positive, GreaterThan, LessThan, LogTransformedInterval, NonTransformedInterval
]

PRIOR_MAP = {}


def register(
    data_model_cls: Type,
    map_fn: Optional[Callable] = None,
):
    """Register a custom prior/constraint mapping from data model to factory function.

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
        PRIOR_MAP[data_model_cls] = fn

        # Also register with the data model unions so Pydantic accepts the type
        from bofire.data_models.priors.constraint import PriorConstraint
        from bofire.data_models.priors.interval import Interval
        from bofire.data_models.priors.prior import Prior

        if issubclass(data_model_cls, Prior):
            from bofire.data_models.priors.api import register_prior

            register_prior(data_model_cls)
        elif issubclass(data_model_cls, (PriorConstraint, Interval)):
            from bofire.data_models.priors.api import register_prior_constraint

            register_prior_constraint(data_model_cls)

        return fn

    if map_fn is not None:
        _register(map_fn)
        return None

    return _register


@register(data_models.NormalPrior)
def map_NormalPrior(
    data_model: data_models.NormalPrior,
    **kwargs,
) -> gpytorch.priors.NormalPrior:
    return gpytorch.priors.NormalPrior(loc=data_model.loc, scale=data_model.scale)


@register(data_models.GammaPrior)
def map_GammaPrior(
    data_model: data_models.GammaPrior,
    **kwargs,
) -> gpytorch.priors.GammaPrior:
    return gpytorch.priors.GammaPrior(
        concentration=data_model.concentration,
        rate=data_model.rate,
    )


@register(data_models.LKJPrior)
def map_LKJPrior(
    data_model: data_models.LKJPrior,
    **kwargs,
) -> gpytorch.priors.LKJPrior:
    return gpytorch.priors.LKJCovariancePrior(
        n=data_model.n_tasks,
        eta=data_model.shape,
        sd_prior=map(data_model.sd_prior),
    )


@register(data_models.LogNormalPrior)
def map_LogNormalPrior(
    data_model: data_models.LogNormalPrior,
    **kwargs,
) -> gpytorch.priors.LogNormalPrior:
    return gpytorch.priors.LogNormalPrior(loc=data_model.loc, scale=data_model.scale)


@register(data_models.DimensionalityScaledLogNormalPrior)
def map_DimensionalityScaledLogNormalPrior(
    data_model: data_models.DimensionalityScaledLogNormalPrior,
    d: int,
) -> gpytorch.priors.LogNormalPrior:
    return gpytorch.priors.LogNormalPrior(
        loc=data_model.loc + math.log(d) * data_model.loc_scaling,
        scale=(data_model.scale**2 + math.log(d) * data_model.scale_scaling) ** 0.5,
    )


@register(data_models.NonTransformedInterval)
def map_NonTransformedInterval(
    data_model: data_models.NonTransformedInterval,
) -> NonTransformedInterval:
    return NonTransformedInterval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=data_model.initial_value,
    )


@register(data_models.LogTransformedInterval)
def map_LogTransformedInterval(
    data_model: data_models.LogTransformedInterval,
) -> LogTransformedInterval:
    return LogTransformedInterval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=data_model.initial_value,
    )


@register(data_models.Positive)
def map_Positive(
    data_model: data_models.Positive,
) -> Positive:
    return Positive()


@register(data_models.GreaterThan)
def map_GreaterThan(
    data_model: data_models.GreaterThan,
) -> GreaterThan:
    return GreaterThan(lower_bound=data_model.lower_bound, transform=None)


@register(data_models.LessThan)
def map_LessThan(
    data_model: data_models.LessThan,
) -> LessThan:
    return LessThan(upper_bound=data_model.upper_bound, transform=None)


def map(
    data_model: Union[data_models.AnyPrior, data_models.AnyPriorConstraint], **kwargs
) -> Union[gpytorch.priors.Prior, Constraint]:
    return PRIOR_MAP[data_model.__class__](data_model, **kwargs)
