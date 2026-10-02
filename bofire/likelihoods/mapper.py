from typing import Callable, Dict, Type

import gpytorch
from botorch.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)

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


def map_PairwiseProbitLikelihood(
    data_model: data_models.PairwiseProbitLikelihood, d: int, **kwargs
) -> PairwiseProbitLikelihood:
    return PairwiseProbitLikelihood()


def map_PairwiseLogitLikelihood(
    data_model: data_models.PairwiseLogitLikelihood, d: int, **kwargs
) -> PairwiseLogitLikelihood:
    return PairwiseLogitLikelihood()


LIKELIHOOD_MAP: Dict[Type[data_models.Likelihood], Callable] = {
    data_models.GaussianLikelihood: map_GaussianLikelihood,
    data_models.PairwiseProbitLikelihood: map_PairwiseProbitLikelihood,
    data_models.PairwiseLogitLikelihood: map_PairwiseLogitLikelihood,
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
