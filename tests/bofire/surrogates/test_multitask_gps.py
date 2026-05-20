import importlib

import gpytorch
import pandas as pd
import pytest
import torch
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Log, Standardize
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.benchmarks.api import MultiTaskHimmelblau
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import MaternKernel, RBFKernel
from bofire.data_models.priors.api import (
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
    else:
        assert surrogate_data.noise_prior == THREESIX_NOISE_PRIOR()
        assert surrogate_data.kernel.lengthscale_prior == THREESIX_LENGTHSCALE_PRIOR()


def test_MultiTask_input_preprocessing():
    # test that if no input_preprocessing_specs are provided, the ordinal encoding is used
    inputs = Inputs(
        features=[ContinuousInput(key="x", bounds=(-1, 1))]
        + [CategoricalTaskInput(key="task_id", categories=["1", "2"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    data_model = MultiTaskGPSurrogate(inputs=inputs, outputs=outputs)
    assert data_model.input_preprocessing_specs == {
        "task_id": CategoricalEncodingEnum.ORDINAL,
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
        "categories": CategoricalEncodingEnum.ORDINAL,
        "task_id": CategoricalEncodingEnum.ORDINAL,
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

    model = surrogates.map(model)
    with pytest.raises(ValueError):
        model.dumps()
    # if task_prior is not None, a warning should be raised
    if task_prior is not None:
        with pytest.warns(UserWarning):
            model.fit(experiments)
    else:
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
    # check that model is composed correctly
    assert isinstance(model.model, MultiTaskGP)
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

    assert (
        large_noise.model.likelihood.noise.item()
        > small_noise.model.likelihood.noise.item()
    )
