import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.encodings.api import OrdinalEncoding
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    AdditiveMapSaasKernel,
    HammingDistanceKernel,
)
from bofire.data_models.surrogates.api import (
    AdditiveMapSaasSingleTaskGPSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
    ScalerEnum,
    SingleTaskGPSurrogate,
)
from bofire.surrogates.botorch import TrainableBotorchSurrogate


def _assert_is_additive_map_saas(model, n_taus: int):
    """A single-task GP built from the components of BoTorch's additive MAP-SAAS GP."""
    assert type(model) is SingleTaskGP
    assert isinstance(model.covar_module, gpytorch.kernels.AdditiveKernel)
    assert len(model.covar_module.kernels) == n_taus
    ((_, _, mean_prior, *_),) = model.mean_module.named_priors()
    assert isinstance(mean_prior, gpytorch.priors.NormalPrior)


class _BotorchAdditiveMapSaas(TrainableBotorchSurrogate):
    """Fits BoTorch's own AdditiveMapSaasSingleTaskGP, as BoFire did before #825."""

    def __init__(self, data_model, **kwargs):
        self.n_taus = data_model.n_taus
        super().__init__(data_model=data_model, **kwargs)

    training_specs: dict = {}

    def _fit_botorch(self, tX, tY, input_transform=None, outcome_transform=None, **kw):
        self.model = AdditiveMapSaasSingleTaskGP(
            train_X=tX,
            train_Y=tY,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            num_taus=self.n_taus,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=50)


def _himmelblau(n: int = 10):
    bench = Himmelblau()
    torch.manual_seed(0)
    samples = bench.domain.inputs.sample(n, seed=0)
    return bench, bench.f(samples, return_complete=True)


def test_additive_map_saas_is_botorchs_model():
    """Built from components, the surrogate fits the same model BoTorch's class does."""
    bench, experiments = _himmelblau()
    data_model = AdditiveMapSaasSingleTaskGPSurrogate(
        inputs=bench.domain.inputs, outputs=bench.domain.outputs
    )

    torch.manual_seed(1)
    ours = surrogates.map(data_model)
    ours.fit(experiments=experiments)
    torch.manual_seed(1)
    reference = _BotorchAdditiveMapSaas(data_model=data_model)
    reference.fit(experiments=experiments)

    assert_frame_equal(
        ours.predict(experiments), reference.predict(experiments), check_exact=True
    )


def test_additive_map_saas_loads_a_dump_of_botorchs_model():
    """Fitted models dumped before #825 hold BoTorch's class and still load."""
    bench, experiments = _himmelblau()
    data_model = AdditiveMapSaasSingleTaskGPSurrogate(
        inputs=bench.domain.inputs, outputs=bench.domain.outputs
    )
    old = _BotorchAdditiveMapSaas(data_model=data_model)
    old.fit(experiments=experiments)

    new = surrogates.map(data_model)
    new.loads(old.dumps())

    assert isinstance(new.model, AdditiveMapSaasSingleTaskGP)
    assert_frame_equal(new.predict(experiments), old.predict(experiments))


def test_AdditiveMapSaasSingleTaskGPSurrogate():
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(10)
    experiments = bench.f(samples, return_complete=True)
    data_model = AdditiveMapSaasSingleTaskGPSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    gp = surrogates.map(data_model)
    gp.fit(experiments=experiments)
    _assert_is_additive_map_saas(gp.model, n_taus=data_model.n_taus)
    dump = gp.dumps()
    gp2 = surrogates.map(data_model=data_model)
    gp2.loads(dump)
    preds = gp.predict(experiments)
    assert preds.shape == (10, 2)
    preds2 = gp.predict(experiments)
    assert_frame_equal(preds, preds2)


def test_EnsembleMapSaasSingleTaskGPSurrogate():
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(10)
    experiments = bench.f(samples, return_complete=True)
    data_model = EnsembleMapSaasSingleTaskGPSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    gp = surrogates.map(data_model)
    gp.fit(experiments=experiments)
    assert isinstance(gp.model, EnsembleMapSaasSingleTaskGP)
    dump = gp.dumps()
    gp2 = surrogates.map(data_model=data_model)
    gp2.loads(dump)
    preds = gp.predict(experiments)
    assert preds.shape == (10, 2)
    preds2 = gp.predict(experiments)
    assert_frame_equal(preds, preds2)


def test_AdditiveMapSaasSingleTaskGPSurrogate_log_output_transform():
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(10)
    experiments = bench.f(samples, return_complete=True)
    data_model = AdditiveMapSaasSingleTaskGPSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        output_scaler=ScalerEnum.LOG,
    )
    gp = surrogates.map(data_model)
    gp.fit(experiments=experiments)
    _assert_is_additive_map_saas(gp.model, n_taus=data_model.n_taus)
    dump = gp.dumps()
    gp2 = surrogates.map(data_model=data_model)
    gp2.loads(dump)
    preds = gp.predict(experiments)
    assert preds.shape == (10, 2)
    preds2 = gp.predict(experiments)
    assert_frame_equal(preds, preds2)


def test_AdditiveMapSaasSingleTaskGPSurrogate_chained_log_output_transform():
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(10)
    experiments = bench.f(samples, return_complete=True)
    data_model = AdditiveMapSaasSingleTaskGPSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        output_scaler=ScalerEnum.CHAINED_LOG_STANDARDIZE,
    )
    gp = surrogates.map(data_model)
    gp.fit(experiments=experiments)
    _assert_is_additive_map_saas(gp.model, n_taus=data_model.n_taus)
    dump = gp.dumps()
    gp2 = surrogates.map(data_model=data_model)
    gp2.loads(dump)
    preds = gp.predict(experiments)
    assert preds.shape == (10, 2)
    preds2 = gp.predict(experiments)
    assert_frame_equal(preds, preds2)


def test_additive_map_saas_kernel_composes_with_other_kernels():
    """As a kernel it can cover part of the inputs, next to a categorical kernel."""
    inputs = Inputs(
        features=[
            *[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(4)],
            CategoricalInput(key="c", categories=["a", "b"]),
        ]
    )
    np.random.seed(0)
    experiments = pd.DataFrame(
        np.random.rand(12, 4), columns=[f"x{i}" for i in range(4)]
    )
    experiments["c"] = np.random.choice(["a", "b"], 12)
    experiments["y"] = experiments.x0 + (experiments.c == "a")
    experiments["valid_y"] = 1
    data_model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        categorical_encodings={"c": OrdinalEncoding()},
        kernel=AdditiveKernel(
            kernels=[
                AdditiveMapSaasKernel(features=[f"x{i}" for i in range(4)]),
                HammingDistanceKernel(features=["c"]),
            ]
        ),
    )

    surrogate = surrogates.map(data_model)
    surrogate.fit(experiments)

    saas, hamming = surrogate.model.covar_module.kernels
    assert saas.kernels[0].base_kernel.active_dims.tolist() == [0, 1, 2, 3]
    assert hamming.active_dims.tolist() == [4]
