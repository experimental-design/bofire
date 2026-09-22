import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import LinearKernel
from bofire.data_models.priors.api import (
    HVARFNER_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    GreaterThan,
)
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    LinearSurrogate,
    ScalerEnum,
    SingleTaskGPSurrogate,
)


def test_LinearSurrogate():
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

    surrogate_data = LinearSurrogate(inputs=inputs, outputs=outputs)
    surrogate_data.noise_constraint = GreaterThan(lower_bound=5e-4)
    surrogate = surrogates.map(surrogate_data)

    assert isinstance(surrogate, surrogates.SingleTaskGPSurrogate)
    assert isinstance(surrogate.kernel, LinearKernel)
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
                LinearSurrogate(inputs=inputs, outputs=Outputs(features=[outputs[0]])),
                LinearSurrogate(inputs=inputs, outputs=Outputs(features=[outputs[1]])),
            ],
        ),
    )


def test_linear_surrogate_is_a_single_task_gp():
    """The preset is a function, so what it returns serializes as a plain GP."""
    inputs = Inputs(features=[ContinuousInput(key="a", bounds=(0, 40))])
    outputs = Outputs(features=[ContinuousOutput(key="c")])

    surrogate_data = LinearSurrogate(inputs=inputs, outputs=outputs)

    assert isinstance(surrogate_data, SingleTaskGPSurrogate)
    assert surrogate_data.type == "SingleTaskGPSurrogate"
    assert surrogate_data.kernel == LinearKernel()
    # the single-task GP search would replace the linear kernel
    assert surrogate_data.hyperconfig is None


def test_linear_surrogate_noise_constraint_matches_the_gp_default():
    """The preset leaves `noise_constraint` alone rather than restating it.

    It used to set `GreaterThan(lower_bound=1e-4)` explicitly, which is exactly what
    `SingleTaskGPSurrogate` already defaults to, so the override changed nothing.
    Leaving it out means pydantic supplies a freshly deep-copied default.
    """
    inputs = Inputs(features=[ContinuousInput(key="a", bounds=(0, 40))])
    outputs = Outputs(features=[ContinuousOutput(key="c")])

    first = LinearSurrogate(inputs=inputs, outputs=outputs)
    second = LinearSurrogate(inputs=inputs, outputs=outputs)

    assert first.noise_constraint == GreaterThan(lower_bound=1e-4)
    assert first.noise_constraint is not second.noise_constraint


def test_linear_surrogate_explicit_arguments_override_the_preset():
    inputs = Inputs(features=[ContinuousInput(key="a", bounds=(0, 40))])
    outputs = Outputs(features=[ContinuousOutput(key="c")])

    surrogate = LinearSurrogate(
        inputs=inputs,
        outputs=outputs,
        noise_prior=HVARFNER_NOISE_PRIOR(),
    )

    assert surrogate.noise_prior == HVARFNER_NOISE_PRIOR()


def test_preset_takes_only_what_it_decides():
    """A field the preset has no opinion on is set on the result, not passed in."""
    inputs = Inputs(features=[ContinuousInput(key="a", bounds=(0, 40))])
    outputs = Outputs(features=[ContinuousOutput(key="c")])

    with pytest.raises(TypeError, match="output_scaler"):
        LinearSurrogate(inputs=inputs, outputs=outputs, output_scaler=ScalerEnum.LOG)

    surrogate = LinearSurrogate(inputs=inputs, outputs=outputs)
    surrogate.output_scaler = ScalerEnum.LOG
    assert surrogate.output_scaler == ScalerEnum.LOG


def test_linear_surrogate_exposes_the_kernel_variance_prior():
    """The kernel is fixed, so its hyperparameters have to be reachable through it."""
    inputs = Inputs(features=[ContinuousInput(key="a", bounds=(0, 40))])
    outputs = Outputs(features=[ContinuousOutput(key="c")])

    surrogate = LinearSurrogate(
        inputs=inputs,
        outputs=outputs,
        variance_prior=THREESIX_SCALE_PRIOR(),
    )

    assert surrogate.kernel == LinearKernel(variance_prior=THREESIX_SCALE_PRIOR())
    # omitting it leaves the kernel's own default
    assert LinearSurrogate(inputs=inputs, outputs=outputs).kernel == LinearKernel()
