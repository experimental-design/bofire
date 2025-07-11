import pytest
from botorch.models.fully_bayesian import (
    FullyBayesianLinearSingleTaskGP,
    SaasFullyBayesianSingleTaskGP,
)
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.surrogates.api import FullyBayesianSingleTaskGPSurrogate


@pytest.mark.parametrize(
    "model_type, expected_class, with_warping",
    [
        ("saas", SaasFullyBayesianSingleTaskGP, False),
        ("linear", FullyBayesianLinearSingleTaskGP, False),
        # ("hvarfner", FullyBayesianSingleTaskGP, False),
        ("saas", SaasFullyBayesianSingleTaskGP, True),
    ],
)
def test_FullyBayesianTaskGPSurrogate(model_type, expected_class, with_warping):
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(5)
    experiments = bench.f(samples, return_complete=True)
    data_model = FullyBayesianSingleTaskGPSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        warmup_steps=32,
        num_samples=16,
        thinning=4,
        model_type=model_type,
        features_to_warp=[
            "x_2",
        ]
        if with_warping
        else [],
    )
    gp = surrogates.map(data_model)
    gp.fit(experiments=experiments)
    assert isinstance(gp.model, expected_class)

    if hasattr(gp.model.pyro_model, "use_input_warping"):
        if with_warping:
            assert gp.model.pyro_model.use_input_warping is True
            assert gp.model.pyro_model.indices == [1]
        else:
            assert gp.model.pyro_model.use_input_warping is False
            assert gp.model.pyro_model.indices is None

    dump = gp.dumps()
    gp2 = surrogates.map(data_model=data_model)
    gp2.loads(dump)
    preds = gp.predict(experiments)
    assert preds.shape == (5, 2)
    preds2 = gp.predict(experiments)
    assert_frame_equal(preds, preds2)
