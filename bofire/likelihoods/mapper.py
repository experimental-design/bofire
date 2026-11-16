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


def register(
    data_model_cls: Type[data_models.Likelihood],
    map_fn: Optional[Callable] = None,
    *,
    single_task: bool = True,
    multi_task: bool = False,
):
    """Register a custom likelihood mapping from data model to factory function.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyLikelihoodDataModel, single_task=True, multi_task=False)
        def map_my_likelihood(data_model, d, **kwargs):
            return MyGpytorchLikelihood(...)

        # Direct call form
        register(
            MyLikelihoodDataModel, map_my_likelihood, single_task=True, multi_task=False
        )

    BoFire only supports fitting via exact marginal log likelihood, so the registered
    likelihood must be a subclass of `gpytorch.likelihoods._GaussianLikelihoodBase`.
    Non-Gaussian likelihoods will fail when the surrogate is fitted.

    Args:
        data_model_cls: The Pydantic data model class.
        map_fn: A callable that takes ``(data_model, d, **kwargs)`` and returns a
            gpytorch likelihood. ``d`` is the number of input dimensions the model
            sees. A multi-task GP also passes ``num_tasks`` and
            ``task_feature_index``, the column holding the task. If not provided,
            returns a decorator.
        single_task: Whether a ``SingleTaskGPSurrogate`` accepts the likelihood.
        multi_task: Whether a ``MultiTaskGPSurrogate`` accepts the likelihood.

    Returns:
        The mapping function (unchanged) when used as a decorator, None otherwise.

    Raises:
        ValueError: If a different likelihood with the same ``type``
            discriminator is already registered.
    """

    def _register(fn: Callable) -> Callable:
        # Register with the data model unions first so a discriminator conflict
        # is raised before the functional map is touched (no partial state).
        data_models.register_likelihood(
            data_model_cls, single_task=single_task, multi_task=multi_task
        )
        LIKELIHOOD_MAP[data_model_cls] = fn

        return fn

    if map_fn is not None:
        _register(map_fn)
        return None

    return _register


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
