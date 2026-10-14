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
from bofire.data_models.priors.api import GammaPrior, GreaterThan
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate


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
        # a prior without a constraint: GPyTorch's positivity, starting at the mode
        ({"lengthscale_prior": GammaPrior(concentration=2, rate=4)}, 0.25),
    ],
)
def test_starting_values(kwargs, start):
    assert _lengthscale(**kwargs) == pytest.approx(start)


def test_training_specs_reach_scipy(monkeypatch):
    import botorch.optim.fit as botorch_fit

    received = []
    scipy_minimize = botorch_fit.scipy_minimize

    def spy(*args, **kwargs):
        received.append(kwargs.get("options"))
        return scipy_minimize(*args, **kwargs)

    monkeypatch.setattr(botorch_fit, "scipy_minimize", spy)
    bench = Himmelblau()
    experiments = bench.f(bench.domain.inputs.sample(10, seed=0), return_complete=True)
    surrogate = surrogates.map(
        SingleTaskGPSurrogate(inputs=bench.domain.inputs, outputs=bench.domain.outputs)
    )
    surrogate.training_specs = {"maxiter": 3}

    surrogate.fit(experiments)

    assert received and all(options["maxiter"] == 3 for options in received)
