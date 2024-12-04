import numpy as np
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import PolynomialKernel
from bofire.data_models.surrogates.api import BotorchSurrogates, PolynomialSurrogate


def test_polynomial_surrogate():
    N_EXPERIMENTS = 10

    inputs = Inputs(
        features=[
            ContinuousInput(key="a", bounds=(0, 40)),
            ContinuousInput(key="b", bounds=(20, 60)),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="c")])

    experiments = inputs.sample(N_EXPERIMENTS)
    experiments["c"] = (
        experiments["a"] * 2.2
        + experiments["b"] * -0.05
        + experiments["b"]
        + np.random.normal(loc=0, scale=5, size=N_EXPERIMENTS)
    )
    experiments["valid_c"] = 1

    surrogate_data = PolynomialSurrogate.from_power(
        power=2,
        inputs=inputs,
        outputs=outputs,
    )
    surrogate = surrogates.map(surrogate_data)

    assert isinstance(surrogate, surrogates.SingleTaskGPSurrogate)
    assert isinstance(surrogate.kernel, PolynomialKernel)

    # check dump
    surrogate.fit(experiments=experiments)
    preds = surrogate.predict(experiments)
    dump = surrogate.dumps()
    surrogate.loads(dump)
    preds2 = surrogate.predict(experiments)
    assert_frame_equal(preds, preds2)


def test_can_define_botorch_surrogate():
    inputs = Inputs(
        features=[
            ContinuousInput(key="a", bounds=(0, 40)),
            ContinuousInput(key="b", bounds=(20, 80)),
        ],
    )
    outputs = [ContinuousOutput(key="c"), ContinuousOutput(key="d")]
    (
        BotorchSurrogates(
            surrogates=[
                PolynomialSurrogate(
                    inputs=inputs,
                    outputs=Outputs(features=[outputs[0]]),
                ),
                PolynomialSurrogate(
                    inputs=inputs,
                    outputs=Outputs(features=[outputs[1]]),
                ),
            ],
        ),
    )
