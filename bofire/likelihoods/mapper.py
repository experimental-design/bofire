from typing import Callable, Dict, Type

import gpytorch

import bofire.data_models.likelihoods.api as data_models
import bofire.priors.api as priors


def map_GaussianLikelihood(
    data_model: data_models.GaussianLikelihood, d: int, **kwargs
) -> gpytorch.likelihoods.GaussianLikelihood:
    return gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=priors.map(data_model.noise_prior, d=d),
        noise_constraint=priors.map(data_model.noise_constraint)
        if data_model.noise_constraint is not None
        else None,
    )


LIKELIHOOD_MAP: Dict[Type[data_models.Likelihood], Callable] = {
    data_models.GaussianLikelihood: map_GaussianLikelihood,
}


def map(
    data_model: data_models.AnyLikelihood, d: int, **kwargs
) -> gpytorch.likelihoods.Likelihood:
    """Build the gpytorch likelihood for a likelihood data model.

    Args:
        data_model: The likelihood to build.
        d: Number of input dimensions the model sees, used by dimensionality-scaled
            priors.

    Returns:
        The gpytorch likelihood.
    """
    return LIKELIHOOD_MAP[data_model.__class__](data_model, d=d, **kwargs)
