import importlib
import math

import gpytorch
import numpy as np
import pandas as pd
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.kernels.positive_index import PositiveIndexKernel
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize
from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from pandas.testing import assert_frame_equal

import bofire.kernels.api as bofire_kernels
import bofire.surrogates.api as surrogates
from bofire.benchmarks.api import MultiTaskHimmelblau
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.encodings.api import OrdinalEncoding
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import MaternKernel, RBFKernel
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    LKJ_PRIOR,
    MBO_LENGTHSCALE_PRIOR,
    MBO_NOISE_PRIOR,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    GammaPrior,
    GreaterThan,
    LogNormalPrior,
)
from bofire.data_models.surrogates.api import MultiTaskGPSurrogate, ScalerEnum
from bofire.data_models.surrogates.scaler import Normalize as NormalizeScaler
from bofire.data_models.surrogates.scaler import Standardize as StandardizeScaler
from bofire.surrogates.botorch import TrainableBotorchSurrogate


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


def test_MultiTaskGPHyperconfig():
    # we test here also the basic trainable
    benchmark = MultiTaskHimmelblau()
    surrogate_data_no_hy = MultiTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        hyperconfig=None,
    )

    with pytest.raises(ValueError, match="No hyperconfig available."):
        surrogate_data_no_hy.update_hyperparameters(
            benchmark.domain.inputs.sample(1).loc[0],
        )
    # test that correct stuff is written
    surrogate_data = MultiTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
    )
    candidate = surrogate_data.hyperconfig.inputs.sample(1).loc[0]
    surrogate_data.update_hyperparameters(candidate)

    assert surrogate_data.kernel.ard == (candidate["ard"] == "True")
    if candidate.kernel == "matern_1.5":
        assert isinstance(surrogate_data.kernel, MaternKernel)
        assert surrogate_data.kernel.nu == 1.5
    elif candidate.kernel == "matern_2.5":
        assert isinstance(surrogate_data.kernel, MaternKernel)
        assert surrogate_data.kernel.nu == 2.5
    else:
        assert isinstance(surrogate_data.kernel, RBFKernel)
    if candidate.prior == "mbo":
        assert surrogate_data.noise_prior == MBO_NOISE_PRIOR()
        assert surrogate_data.kernel.lengthscale_prior == MBO_LENGTHSCALE_PRIOR()
    elif candidate.prior == "threesix":
        assert surrogate_data.noise_prior == THREESIX_NOISE_PRIOR()
        assert surrogate_data.kernel.lengthscale_prior == THREESIX_LENGTHSCALE_PRIOR()
    else:
        assert surrogate_data.noise_prior == HVARFNER_NOISE_PRIOR()
        assert surrogate_data.kernel.lengthscale_prior == HVARFNER_LENGTHSCALE_PRIOR()


def test_MultiTask_input_preprocessing():
    # test that if no input_preprocessing_specs are provided, the ordinal encoding is used
    inputs = Inputs(
        features=[ContinuousInput(key="x", bounds=(-1, 1))]
        + [CategoricalTaskInput(key="task_id", categories=["1", "2"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    data_model = MultiTaskGPSurrogate(inputs=inputs, outputs=outputs)
    assert data_model.input_preprocessing_specs == {
        "task_id": OrdinalEncoding(),
    }

    # test that if we have a categorical input, one-hot encoding is correctly applied
    inputs = Inputs(
        features=[ContinuousInput(key="x", bounds=(-1, 1))]
        + [CategoricalInput(key="categories", categories=["1", "2"])]
        + [CategoricalTaskInput(key="task_id", categories=["1", "2"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    data_model = MultiTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    assert data_model.input_preprocessing_specs == {
        "categories": OrdinalEncoding(),
        "task_id": OrdinalEncoding(),
    }


@pytest.mark.parametrize(
    "kernel, scaler, output_scaler, task_prior",
    [
        (RBFKernel(ard=True), NormalizeScaler(), ScalerEnum.STANDARDIZE, None),
        (RBFKernel(ard=False), StandardizeScaler(), ScalerEnum.STANDARDIZE, None),
        (RBFKernel(ard=False), None, ScalerEnum.IDENTITY, LKJ_PRIOR()),
        (RBFKernel(ard=False), StandardizeScaler(), ScalerEnum.LOG, None),
        (
            RBFKernel(ard=False),
            StandardizeScaler(),
            ScalerEnum.CHAINED_LOG_STANDARDIZE,
            None,
        ),
    ],
)
def test_MultiTaskGPModel(kernel, scaler, output_scaler, task_prior):
    benchmark = MultiTaskHimmelblau()
    inputs = benchmark.domain.inputs
    outputs = benchmark.domain.outputs
    # Sample both tasks to ensure both are present in training data
    experiments_task1 = benchmark.f(
        inputs.sample(5, seed=42).assign(task_id="task_1"), return_complete=True
    )
    experiments_task2 = benchmark.f(
        inputs.sample(5, seed=43).assign(task_id="task_2"), return_complete=True
    )
    experiments = pd.concat([experiments_task1, experiments_task2], ignore_index=True)

    model = MultiTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        scaler=scaler,
        output_scaler=output_scaler,
        kernel=kernel,
        task_prior=task_prior,
    )

    # a task_prior never reaches the model, and building it says so
    if task_prior is not None:
        with pytest.warns(UserWarning):
            model = surrogates.map(model)
    else:
        model = surrogates.map(model)
    with pytest.raises(ValueError):
        model.dumps()
    model.fit(experiments)
    # check that the active_dims are set correctly
    assert torch.allclose(
        model.model.covar_module.kernels[0].active_dims,
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.allclose(
        model.model.covar_module.kernels[0].active_dims,
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.allclose(
        model.model.covar_module.kernels[1].active_dims,
        torch.tensor([2], dtype=torch.long),
    )
    # dump the model
    dump = model.dumps()
    # make predictions
    samples = inputs.sample(5)
    preds = model.predict(samples)
    assert preds.shape == (5, 2)
    # check that model is composed correctly: a single-task GP with per-task components
    assert type(model.model) is SingleTaskGP
    assert isinstance(model.model.likelihood, HadamardGaussianLikelihood)
    if output_scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.outcome_transform, Standardize)
    elif output_scaler == ScalerEnum.LOG:
        assert isinstance(model.model.outcome_transform, Log)
    elif output_scaler == ScalerEnum.CHAINED_LOG_STANDARDIZE:
        assert isinstance(model.model.outcome_transform, ChainedOutcomeTransform)
    elif output_scaler == ScalerEnum.IDENTITY:
        assert not hasattr(model.model, "outcome_transform")
    if isinstance(scaler, NormalizeScaler):
        assert isinstance(model.model.input_transform, Normalize)
    elif isinstance(scaler, StandardizeScaler):
        assert isinstance(model.model.input_transform, InputStandardize)
    else:
        assert not hasattr(model.model, "input_transform")
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MultiTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
        output_scaler=output_scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)


def test_MultiTaskGPModel_noise_constraint():
    benchmark = MultiTaskHimmelblau()
    inputs = benchmark.domain.inputs
    outputs = benchmark.domain.outputs
    experiments_task1 = benchmark.f(
        inputs.sample(5, seed=42).assign(task_id="task_1"), return_complete=True
    )
    experiments_task2 = benchmark.f(
        inputs.sample(5, seed=43).assign(task_id="task_2"), return_complete=True
    )
    experiments = pd.concat([experiments_task1, experiments_task2], ignore_index=True)

    model = MultiTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        noise_constraint=GreaterThan(lower_bound=5e-4),
    )
    model = surrogates.map(model)
    model.fit(experiments)
    lower_bound = float(
        model.model.likelihood.noise_covar.raw_noise_constraint.lower_bound
    )
    assert lower_bound >= 5e-4


# --- Regression tests for issue #762: noise_prior registration on MultiTaskGP ---


def _get_registered_noise_prior(model):
    """Return the noise prior actually registered in ``model.likelihood`` (the
    entry MLL reads from ``_priors`` via ``named_priors()``). Returns ``None``
    if not registered.
    """
    priors = {n: p for n, _, p, _, _ in model.likelihood.named_priors()}
    return priors.get("noise_covar.noise_prior")


def _multi_task_experiments():
    benchmark = MultiTaskHimmelblau()
    inputs = benchmark.domain.inputs
    exp1 = benchmark.f(
        inputs.sample(5, seed=42).assign(task_id="task_1"), return_complete=True
    )
    exp2 = benchmark.f(
        inputs.sample(5, seed=43).assign(task_id="task_2"), return_complete=True
    )
    return benchmark, pd.concat([exp1, exp2], ignore_index=True)


def test_noise_prior_registered_for_multi_task_gp():
    torch.manual_seed(42)
    benchmark, experiments = _multi_task_experiments()

    surrogate = surrogates.map(
        MultiTaskGPSurrogate(
            inputs=benchmark.domain.inputs,
            outputs=benchmark.domain.outputs,
            noise_prior=GammaPrior(concentration=1.1, rate=0.001),
        )
    )
    surrogate.fit(experiments)

    prior = _get_registered_noise_prior(surrogate.model)
    assert isinstance(prior, gpytorch.priors.GammaPrior), (
        f"User-supplied GammaPrior must be in the likelihood's _priors registry "
        f"(got {type(prior).__name__})"
    )


def test_noise_prior_directional_effect_on_multi_task_gp():
    """Tiny-noise prior vs large-noise prior should pull the fitted noise in
    opposite directions. Before the fix, fitted noise was identical regardless
    of the user-supplied prior.
    """
    torch.manual_seed(42)
    benchmark, experiments = _multi_task_experiments()

    small_noise = surrogates.map(
        MultiTaskGPSurrogate(
            inputs=benchmark.domain.inputs,
            outputs=benchmark.domain.outputs,
            noise_prior=LogNormalPrior(loc=-8.0, scale=0.1),  # mode ~ 3e-4
        )
    )
    large_noise = surrogates.map(
        MultiTaskGPSurrogate(
            inputs=benchmark.domain.inputs,
            outputs=benchmark.domain.outputs,
            noise_prior=GammaPrior(concentration=1.1, rate=0.001),  # mode = 100
        )
    )
    small_noise.fit(experiments)
    large_noise.fit(experiments)

    # the noise is learned per task; every task follows its prior
    assert torch.all(
        large_noise.model.likelihood.noise > small_noise.model.likelihood.noise
    )


class _BotorchMultiTaskGP(TrainableBotorchSurrogate):
    """Fits BoTorch's own MultiTaskGP with its per-task defaults, for comparison."""

    def __init__(self, data_model, **kwargs):
        self.kernel = data_model.kernel
        self.task_key = data_model.inputs.get_keys(CategoricalTaskInput)[0]
        self.n_tasks = len(data_model.inputs.get_by_key(self.task_key).categories)
        super().__init__(data_model=data_model, **kwargs)

    training_specs: dict = {}

    def _fit_botorch(self, tX, tY, input_transform=None, outcome_transform=None, **kw):
        (task_index,) = self.get_feature_indices([self.task_key])
        self.model = MultiTaskGP(
            train_X=tX,
            train_Y=tY,
            task_feature=task_index,
            covar_module=bofire_kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=[i for i in range(tX.shape[-1]) if i != task_index],
                features_to_idx_mapper=self.get_feature_indices,
            ),
            task_covar_prior=None,
            all_tasks=list(range(self.n_tasks)),
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)


def _botorch_default_data_model(benchmark):
    # BoTorch starts the noise at its prior's mode; match that for the comparison
    return MultiTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        noise_prior=LogNormalPrior(loc=-4.0, scale=1.0),
        noise_constraint=GreaterThan(lower_bound=1e-4, initial_value=math.exp(-5.0)),
    )


def test_multi_task_gp_is_botorchs_model():
    """From components, the surrogate fits what BoTorch's MultiTaskGP fits."""
    benchmark, experiments = _multi_task_experiments()
    data_model = _botorch_default_data_model(benchmark)

    torch.manual_seed(1)
    ours = surrogates.map(data_model)
    ours.fit(experiments)
    torch.manual_seed(1)
    reference = _BotorchMultiTaskGP(data_model=data_model)
    reference.fit(experiments)

    assert_frame_equal(
        ours.predict(experiments), reference.predict(experiments), check_exact=True
    )


def test_multi_task_gp_components():
    benchmark, experiments = _multi_task_experiments()
    surrogate = surrogates.map(
        MultiTaskGPSurrogate(
            inputs=benchmark.domain.inputs, outputs=benchmark.domain.outputs
        )
    )
    surrogate.fit(experiments)

    base, task = surrogate.model.covar_module.kernels
    # x_1, x_2 are columns 0 and 1; the task code is column 2
    assert base.active_dims.tolist() == [0, 1]
    assert task.active_dims.tolist() == [2]
    assert isinstance(task, PositiveIndexKernel)
    assert task.covar_factor.shape[-1] == 2  # full rank over the two tasks
    assert list(task.named_priors()) == []
    assert isinstance(
        surrogate.model.mean_module.multitask_mean, gpytorch.means.MultitaskMean
    )
    assert surrogate.model.likelihood.noise.shape[-1] == 2


def test_multi_task_gp_predicts_a_task_without_observations():
    """A task that has no data yet is predicted from the prior instead of raising."""
    benchmark, experiments = _multi_task_experiments()
    only_task_1 = experiments[experiments.task_id == "task_1"]
    surrogate = surrogates.map(
        MultiTaskGPSurrogate(
            inputs=benchmark.domain.inputs, outputs=benchmark.domain.outputs
        )
    )
    surrogate.fit(only_task_1)

    preds = surrogate.predict(experiments.assign(task_id="task_2"))

    assert preds.notna().all().all()


def test_multi_task_gp_learns_noise_per_task():
    """Two tasks measured with very different precision get their own noise levels.

    A smooth response and many points keep the kernel from explaining the noise away
    as structure, so the fitted noise can be compared with the truth.
    """
    rng = np.random.default_rng(0)
    n = 40
    x = rng.uniform(0, 1, 2 * n)
    task = np.array(["quiet"] * n + ["noisy"] * n)
    sd = np.where(task == "noisy", 0.5, 0.01)
    experiments = pd.DataFrame(
        {"x": x, "task": task, "y": np.sin(6 * x) + rng.normal(0, sd), "valid_y": 1}
    )
    surrogate = surrogates.map(
        MultiTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x", bounds=(0, 1)),
                    CategoricalTaskInput(key="task", categories=["quiet", "noisy"]),
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
        )
    )
    torch.manual_seed(0)
    surrogate.fit(experiments)

    quiet, noisy = surrogate.model.likelihood.noise.tolist()
    assert noisy > 100 * quiet


def test_multi_task_gp_loads_a_dump_of_botorchs_model():
    """Fitted models dumped before #825 hold BoTorch's MultiTaskGP and still load."""
    benchmark, experiments = _multi_task_experiments()
    data_model = MultiTaskGPSurrogate(
        inputs=benchmark.domain.inputs, outputs=benchmark.domain.outputs
    )
    old = _BotorchMultiTaskGP(data_model=data_model)
    old.fit(experiments)

    new = surrogates.map(data_model)
    new.loads(old.dumps())

    assert isinstance(new.model, MultiTaskGP)
    assert_frame_equal(new.predict(experiments), old.predict(experiments))
