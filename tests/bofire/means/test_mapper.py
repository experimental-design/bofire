import gpytorch
import pytest
from botorch.models.map_saas import get_mean_module_with_normal_prior

import bofire.means.api as means
from bofire.data_models.means.api import ConstantMean, TaskwiseMean
from bofire.data_models.priors.api import NonTransformedInterval, NormalPrior


def _describe(mean: gpytorch.means.ConstantMean) -> tuple:
    ((_, _, prior, *_),) = mean.named_priors()
    constraint = mean.raw_constant_constraint
    return (
        type(prior).__name__,
        float(prior.loc),
        float(prior.scale),
        constraint.enforced,
        float(constraint.lower_bound),
        float(constraint.upper_bound),
        constraint._transform,
        float(constraint.initial_value),
    )


def test_default_constant_mean_is_unconstrained():
    """The default matches what a BoTorch single-task GP builds when given no mean."""
    ours = means.map(ConstantMean(), d=3)

    assert isinstance(ours, gpytorch.means.ConstantMean)
    assert list(ours.named_priors()) == []


@pytest.mark.parametrize("taskwise", [False, True])
def test_constant_mean_with_prior_and_constraint_is_botorchs(taskwise):
    """Normal(0, 1) bounded to [-10, 10] is the mean of BoTorch's MAP-SAAS models."""
    mean = ConstantMean(
        prior=NormalPrior(loc=0.0, scale=1.0),
        constraint=NonTransformedInterval(lower_bound=-10.0, upper_bound=10.0),
    )
    ours = means.map(
        TaskwiseMean(mean=mean) if taskwise else mean,
        d=3,
        num_tasks=2,
    )

    if taskwise:
        assert isinstance(ours, gpytorch.means.MultitaskMean)
        task_means = ours.base_means
        assert len(task_means) == 2
        assert task_means[0].raw_constant is not task_means[1].raw_constant
    else:
        task_means = [ours]
    for task_mean in task_means:
        assert _describe(task_mean) == _describe(get_mean_module_with_normal_prior())
