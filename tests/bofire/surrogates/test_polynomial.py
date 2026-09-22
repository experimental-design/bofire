import numpy as np
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import PolynomialKernel
from bofire.data_models.priors.api import THREESIX_SCALE_PRIOR, GreaterThan
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    PolynomialSurrogate,
    SingleTaskGPSurrogate,
)


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

    surrogate_data = PolynomialSurrogate(
        inputs=inputs,
        outputs=outputs,
        power=2,
    )
    surrogate_data.noise_constraint = GreaterThan(lower_bound=5e-4)
    surrogate = surrogates.map(surrogate_data)

    assert isinstance(surrogate, surrogates.SingleTaskGPSurrogate)
    assert isinstance(surrogate.kernel, PolynomialKernel)
    assert surrogate.noise_constraint is not None

    # check dump
    surrogate.fit(experiments=experiments)
    lower_bound = float(
        surrogate.model.likelihood.noise_covar.raw_noise_constraint.lower_bound
    )
    assert lower_bound >= 5e-4
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


def test_polynomial_surrogate_is_a_single_task_gp():
    """The preset is a function, so what it returns serializes as a plain GP."""
    inputs = Inputs(features=[ContinuousInput(key="a", bounds=(0, 40))])
    outputs = Outputs(features=[ContinuousOutput(key="c")])

    surrogate_data = PolynomialSurrogate(inputs=inputs, outputs=outputs, power=3)

    assert isinstance(surrogate_data, SingleTaskGPSurrogate)
    assert surrogate_data.type == "SingleTaskGPSurrogate"
    assert surrogate_data.kernel == PolynomialKernel(power=3)
    # the single-task GP search would replace the polynomial kernel
    assert surrogate_data.hyperconfig is None


def test_polynomial_surrogate_exposes_the_kernel_offset_prior():
    """The kernel is fixed, so its hyperparameters have to be reachable through it."""
    inputs = Inputs(features=[ContinuousInput(key="a", bounds=(0, 40))])
    outputs = Outputs(features=[ContinuousOutput(key="c")])

    surrogate = PolynomialSurrogate(
        inputs=inputs,
        outputs=outputs,
        power=3,
        offset_prior=THREESIX_SCALE_PRIOR(),
    )

    assert surrogate.kernel == PolynomialKernel(
        power=3, offset_prior=THREESIX_SCALE_PRIOR()
    )
    # omitting it leaves the kernel's own default
    assert PolynomialSurrogate(inputs=inputs, outputs=outputs).kernel == (
        PolynomialKernel(power=2)
    )
