from typing import Callable, Dict, Type

import gpytorch

import bofire.data_models.means.api as data_models
import bofire.priors.api as priors


def map_ConstantMean(
    data_model: data_models.ConstantMean, d: int, **kwargs
) -> gpytorch.means.ConstantMean:
    return gpytorch.means.ConstantMean(
        constant_prior=priors.map(data_model.prior, d=d)
        if data_model.prior is not None
        else None,
        constant_constraint=gpytorch.constraints.Interval(
            *data_model.bounds, transform=None, initial_value=0.0
        )
        if data_model.bounds is not None
        else None,
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
