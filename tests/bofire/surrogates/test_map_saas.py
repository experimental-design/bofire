from botorch.models.map_saas import (
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
)
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.surrogates.api import (
    AdditiveMapSaasSingleTaskGPSurrogate,
    EnsembleMapSaasSingleTaskGPSurrogate,
    ScalerEnum,
)


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
    assert isinstance(gp.model, AdditiveMapSaasSingleTaskGP)
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
    assert isinstance(gp.model, AdditiveMapSaasSingleTaskGP)
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
    assert isinstance(gp.model, AdditiveMapSaasSingleTaskGP)
    dump = gp.dumps()
    gp2 = surrogates.map(data_model=data_model)
    gp2.loads(dump)
    preds = gp.predict(experiments)
    assert preds.shape == (10, 2)
    preds2 = gp.predict(experiments)
    assert_frame_equal(preds, preds2)
