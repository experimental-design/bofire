from typing import Callable, Dict, Optional, Tuple, Type

import gpytorch

import bofire.data_models.means.api as data_models
import bofire.priors.api as priors


def map_ConstantMean(
    data_model: data_models.ConstantMean, d: int, **kwargs
) -> gpytorch.means.ConstantMean:
    prior = priors.map(data_model.prior, d=d) if data_model.prior is not None else None
    mode = _mode(prior)
    mean = gpytorch.means.ConstantMean(
        constant_prior=prior,
        constant_constraint=_bounds(data_model.bounds, mode)
        if data_model.bounds is not None
        else None,
    )
    if data_model.bounds is None and mode is not None:
        # unbounded, so the fit starts at the prior's mode
        mean.initialize(constant=mode)
    return mean


def _mode(prior: Optional[gpytorch.priors.Prior]) -> Optional[float]:
    """The mode of the prior on the constant, if it has one."""
    if prior is None:
        return None
    try:
        return float(prior.mode)
    except (NotImplementedError, AttributeError):
        return None


def _bounds(
    bounds: Tuple[float, float], mode: Optional[float]
) -> gpytorch.constraints.Interval:
    """The bounds on the constant, starting at the prior's mode moved into them, or
    at their middle without one."""
    lower, upper = bounds
    start = (lower + upper) / 2 if mode is None else min(max(mode, lower), upper)
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
