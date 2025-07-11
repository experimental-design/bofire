import math
from typing import Union

import gpytorch
from botorch.utils.constraints import NonTransformedInterval

import bofire.data_models.priors.api as data_models


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
    return gpytorch.priors.LogNormalPrior(
        loc=data_model.loc + math.log(d) * data_model.loc_scaling,
        scale=(data_model.scale**2 + math.log(d) * data_model.scale_scaling) ** 0.5,
    )


def map_NonTransformedInterval(
    data_model: data_models.NonTransformedInterval,
) -> NonTransformedInterval:
    return NonTransformedInterval(
        lower_bound=data_model.lower_bound,
        upper_bound=data_model.upper_bound,
        initial_value=data_model.initial_value,
    )


PRIOR_MAP = {
    data_models.NormalPrior: map_NormalPrior,
    data_models.GammaPrior: map_GammaPrior,
    data_models.LKJPrior: map_LKJPrior,
    data_models.LogNormalPrior: map_LogNormalPrior,
    data_models.DimensionalityScaledLogNormalPrior: map_DimensionalityScaledLogNormalPrior,
    data_models.NonTransformedInterval: map_NonTransformedInterval,
}


def map(
    data_model: Union[data_models.AnyPrior, data_models.AnyPriorConstraint], **kwargs
) -> Union[gpytorch.priors.Prior, gpytorch.constraints.Interval]:
    return PRIOR_MAP[data_model.__class__](data_model, **kwargs)
