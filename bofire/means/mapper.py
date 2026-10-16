from typing import Callable, Dict, Type

import gpytorch

import bofire.data_models.means.api as data_models
import bofire.priors.api as priors


def map_ConstantMean(
    data_model: data_models.ConstantMean, d: int, **kwargs
) -> gpytorch.means.ConstantMean:
    prior = priors.map(data_model.prior, d=d) if data_model.prior is not None else None
    mean = gpytorch.means.ConstantMean(
        constant_prior=prior,
        constant_constraint=priors.map_constraint(data_model.constraint, prior=prior)
        if data_model.constraint is not None
        else None,
    )
    if data_model.constraint is None and prior is not None:
        # unconstrained, so the fit starts at the prior's mode
        mean.initialize(constant=priors.mode(prior))
    return mean


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
