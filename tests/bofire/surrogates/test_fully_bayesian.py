import pytest
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.surrogates.api import SaasSingleTaskGPSurrogate


def test_SaasSingleTaskGPSurrogate_invalid_thinning():
    bench = Himmelblau()
    with pytest.raises(ValueError):
        SaasSingleTaskGPSurrogate(
            inputs=bench.domain.inputs,
            outputs=bench.domain.outputs,
            warmup_steps=32,
            num_samples=16,
            thinning=18,
        )


def test_SaasSingleTaskGPSurrogate():
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(10)
    experiments = bench.f(samples, return_complete=True)
    data_model = SaasSingleTaskGPSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        warmup_steps=32,
        num_samples=16,
        thinning=4,
    )
    gp = surrogates.map(data_model)
    gp.fit(experiments=experiments)
    dump = gp.dumps()
    gp2 = surrogates.map(data_model=data_model)
    gp2.loads(dump)
    preds = gp.predict(experiments)
    assert preds.shape == (10, 2)
    preds2 = gp.predict(experiments)
    assert_frame_equal(preds, preds2)
