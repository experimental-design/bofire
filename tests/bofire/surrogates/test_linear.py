import numpy as np

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import LinearKernel
from bofire.data_models.surrogates.api import LinearSurrogate


def test_LinearSurrogate():
    N_EXPERIMENTS = 10

    inputs = Inputs(
        features=[
            ContinuousInput(key="a", bounds=(0, 40)),
            ContinuousInput(key="b", bounds=(20, 60)),
        ]
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

    surrogate_data = LinearSurrogate(inputs=inputs, outputs=outputs)
    surrogate = surrogates.map(surrogate_data)

    assert isinstance(surrogate, surrogates.SingleTaskGPSurrogate)
    assert isinstance(surrogate.kernel, LinearKernel)
