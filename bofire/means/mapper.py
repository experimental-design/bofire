from typing import Callable, Dict, Type

import gpytorch

import bofire.data_models.means.api as data_models
import bofire.priors.api as priors


def map_ConstantMean(
    data_model: data_models.ConstantMean, d: int, **kwargs
) -> gpytorch.means.ConstantMean:
    mean = gpytorch.means.ConstantMean(
        constant_prior=priors.map(data_model.prior, d=d)
        if data_model.prior is not None
        else None,
        constant_constraint=_bounds(data_model, d)
        if data_model.bounds is not None
        else None,
    )
    if data_model.bounds is None and data_model.prior is not None:
        # unbounded, so the fit starts at the prior's mode where it has one
        mode = priors.map(data_model.prior, d=d)
        try:
            mean.initialize(constant=float(mode.mode))
        except (NotImplementedError, AttributeError):
            pass
    return mean


def _bounds(
    data_model: data_models.ConstantMean, d: int
) -> gpytorch.constraints.Interval:
    """The bounds on the constant, starting at the prior's mode moved into them, or
    at their middle without a prior."""
    assert data_model.bounds is not None
    lower, upper = data_model.bounds
    start = (lower + upper) / 2
    if data_model.prior is not None:
        try:
            mode = float(priors.map(data_model.prior, d=d).mode)
            start = min(max(mode, lower), upper)
        except (NotImplementedError, AttributeError):
            pass
    return gpytorch.constraints.Interval(
        lower, upper, transform=None, initial_value=start
    )


MEAN_MAP: Dict[Type[data_models.Mean], Callable] = {
    data_models.ConstantMean: map_ConstantMean,
}


def map(data_model: data_models.AnyMean, d: int, **kwargs) -> gpytorch.means.Mean:
    """Build the gpytorch mean module for a mean data model.

    Args:
        data_model: The mean to build.
        d: Number of input dimensions the model sees, used by dimensionality-scaled
            priors.

    Returns:
        The gpytorch mean module.
    """
    return MEAN_MAP[data_model.__class__](data_model, d=d, **kwargs)
