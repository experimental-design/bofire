import copy
import math

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.kernels.api import RBFKernel
from bofire.data_models.priors.api import (
    GammaPrior,
    GreaterThan,
    LogTransformedInterval,
    SmoothedBoxPrior,
)
from bofire.data_models.surrogates.api import LBFGSB, SingleTaskGPSurrogate


def test_default_single_task_gp_is_botorchs():
    """Same bounds and starting values, so the same fitted model as BoTorch's default."""
    bench = Himmelblau()
    experiments = bench.f(bench.domain.inputs.sample(15, seed=0), return_complete=True)
    surrogate = surrogates.map(
        SingleTaskGPSurrogate(inputs=bench.domain.inputs, outputs=bench.domain.outputs)
    )
    torch.manual_seed(1)
    surrogate.fit(experiments)
    ours = surrogate.model

    targets, _ = copy.deepcopy(ours.outcome_transform).untransform(
        ours.train_targets.unsqueeze(-1)
    )
    torch.manual_seed(1)
    reference = SingleTaskGP(
        ours.input_transform.untransform(ours.train_inputs[0]),
        targets,
        input_transform=copy.deepcopy(ours.input_transform),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(reference.likelihood, reference))

    X = torch.rand(20, 2, dtype=torch.double) * 12 - 6
    ours.eval()
    reference.eval()
    with torch.no_grad():
        a, b = ours.posterior(X), reference.posterior(X)
    assert torch.equal(a.mean, b.mean)
    assert torch.equal(a.variance, b.variance)


def _lengthscale(**kwargs) -> float:
    kernel = kernels.map(
        RBFKernel(**kwargs),
        batch_shape=torch.Size(),
        active_dims=[0],
        features_to_idx_mapper=None,
    )
    return float(kernel.lengthscale)


@pytest.mark.parametrize(
    "kwargs, start",
    [
        # no prior, bound without a transform: inside the bound
        ({"lengthscale_constraint": GreaterThan(lower_bound=0.5)}, 0.5 + math.log(2)),
        # a starting value on the constraint is kept
        (
            {
                "lengthscale_constraint": GreaterThan(lower_bound=0.5, initial_value=2),
                "lengthscale_prior": GammaPrior(concentration=2, rate=4),
            },
            2.0,
        ),
        # the prior's mode (0.25), here below the bound, moved onto it
        (
            {
                "lengthscale_constraint": GreaterThan(lower_bound=0.5),
                "lengthscale_prior": GammaPrior(concentration=2, rate=4),
            },
            0.5,
        ),
        # the prior's mode inside the bound
        (
            {
                "lengthscale_constraint": GreaterThan(lower_bound=0.1),
                "lengthscale_prior": GammaPrior(concentration=2, rate=4),
            },
            0.25,
        ),
        # a log-transformed interval that excludes 1: its middle on the log scale
        (
            {
                "lengthscale_constraint": LogTransformedInterval(
                    lower_bound=2.0, upper_bound=8.0
                )
            },
            4.0,
        ),
        # a smoothed box has no single mode: the middle of the box
        (
            {
                "lengthscale_constraint": GreaterThan(lower_bound=0.05),
                "lengthscale_prior": SmoothedBoxPrior(
                    lower_bound=0.1, upper_bound=1.0, sigma=0.01
                ),
            },
            0.55,
        ),
        # a prior without a constraint: GPyTorch's positivity, starting at the mode
        ({"lengthscale_prior": GammaPrior(concentration=2, rate=4)}, 0.25),
    ],
)
def test_starting_values(kwargs, start):
    assert _lengthscale(**kwargs) == pytest.approx(start)


@pytest.mark.parametrize(
    "optimizer, options",
    [
        (LBFGSB(), {}),
        (LBFGSB(maxiter=3, gtol=1e-3, max_attempts=2), {"maxiter": 3, "gtol": 1e-3}),
    ],
)
def test_optimizer_reaches_the_fit(monkeypatch, optimizer, options):
    import bofire.surrogates.single_task_gp as single_task_gp

    received = []
    fit = single_task_gp.fit_gpytorch_mll

    def spy(mll, **kwargs):
        received.append(kwargs)
        return fit(mll, **kwargs)

    monkeypatch.setattr(single_task_gp, "fit_gpytorch_mll", spy)
    bench = Himmelblau()
    experiments = bench.f(bench.domain.inputs.sample(10, seed=0), return_complete=True)
    surrogate = surrogates.map(
        SingleTaskGPSurrogate(
            inputs=bench.domain.inputs,
            outputs=bench.domain.outputs,
            optimizer=optimizer,
        )
    )

    surrogate.fit(experiments)

    assert received[0]["optimizer_kwargs"] == {"options": options}
    assert received[0]["max_attempts"] == optimizer.max_attempts
