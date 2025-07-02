from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.surrogates.api import AdditiveMapSaasSingleTaskGPSurrogate


def test_AdditiveMapSaasSingleTaskGPSurrogate():
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(10)
    experiments = bench.f(samples, return_complete=True)
    data_model = AdditiveMapSaasSingleTaskGPSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
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
