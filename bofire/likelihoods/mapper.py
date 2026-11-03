from typing import Callable, Dict, Optional, Tuple, Type

import gpytorch
from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood

import bofire.data_models.likelihoods.api as data_models
import bofire.priors.api as priors


def _noise(
    data_model: data_models.GaussianLikelihood, d: int
) -> Tuple[Optional[gpytorch.priors.Prior], Optional[gpytorch.constraints.Interval]]:
    """The noise prior and constraint of a Gaussian likelihood, as gpytorch objects."""
    noise_prior = (
        priors.map(data_model.noise_prior, d=d)
        if data_model.noise_prior is not None
        else None
    )
    noise_constraint = (
        priors.map_constraint(data_model.noise_constraint, prior=noise_prior)
        if data_model.noise_constraint is not None
        else None
    )
    return noise_prior, noise_constraint


def map_GaussianLikelihood(
    data_model: data_models.GaussianLikelihood, d: int, **kwargs
) -> gpytorch.likelihoods.GaussianLikelihood:
    noise_prior, noise_constraint = _noise(data_model, d=d)
    return gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=noise_prior, noise_constraint=noise_constraint
    )


def map_TaskwiseLikelihood(
    data_model: data_models.TaskwiseLikelihood,
    d: int,
    num_tasks: int,
    task_feature_index: int,
    **kwargs,
) -> HadamardGaussianLikelihood:
    noise_prior, noise_constraint = _noise(data_model.likelihood, d=d)
    # one noise level per task, read from the task column of the inputs
    return HadamardGaussianLikelihood(
        num_tasks=num_tasks,
        noise_prior=noise_prior,
        noise_constraint=noise_constraint,
        task_feature_index=task_feature_index,
    )


LIKELIHOOD_MAP: Dict[Type[data_models.Likelihood], Callable] = {
    data_models.GaussianLikelihood: map_GaussianLikelihood,
    # the base likelihood: BoTorch's RobustRelevancePursuitSingleTaskGP wraps it into
    # its SparseOutlierGaussianLikelihood, which needs the number of training points
    data_models.SparseOutlierGaussianLikelihood: map_GaussianLikelihood,
    data_models.TaskwiseLikelihood: map_TaskwiseLikelihood,
}


def map(
    data_model: data_models.Likelihood, d: int, **kwargs
) -> gpytorch.likelihoods.Likelihood:
    """Build the gpytorch likelihood for a likelihood data model.

    Args:
        data_model: The likelihood to build.
        d: Number of input dimensions the model sees, used by dimensionality-scaled
            priors.
        **kwargs: Context a likelihood may need: a `TaskwiseLikelihood` needs
            `num_tasks` and `task_feature_index`, the column holding the task.

    Returns:
        The gpytorch likelihood.
    """
    return LIKELIHOOD_MAP[data_model.__class__](data_model, d=d, **kwargs)
