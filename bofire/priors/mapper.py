import gpytorch

import bofire.data_models.priors.api as data_models


def map_NormalPrior(data_model: data_models.NormalPrior) -> gpytorch.priors.NormalPrior:
    return gpytorch.priors.NormalPrior(loc=data_model.loc, scale=data_model.scale)


def map_GammaPrior(data_model: data_models.GammaPrior) -> gpytorch.priors.GammaPrior:
    return gpytorch.priors.GammaPrior(
        concentration=data_model.concentration, rate=data_model.rate
    )


PRIOR_MAP = {
    data_models.NormalPrior: map_NormalPrior,
    data_models.GammaPrior: map_GammaPrior,
}


def map(
    data_model: data_models.AnyPrior,
) -> gpytorch.priors.Prior:
    return PRIOR_MAP[data_model.__class__](data_model)
