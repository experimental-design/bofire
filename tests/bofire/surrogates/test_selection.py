import numpy as np
import pandas as pd

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import LinearKernel, MaternKernel, ScaleKernel
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
)
from bofire.data_models.surrogates.api import SelectionSurrogate, SingleTaskGPSurrogate


INPUTS = Inputs(
    features=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(2)]
)
OUTPUTS = Outputs(features=[ContinuousOutput(key="y")])


def _experiments(n: int) -> pd.DataFrame:
    X = np.random.default_rng(0).random((n, 2))
    return pd.DataFrame(
        {
            "x0": X[:, 0],
            "x1": X[:, 1],
            "y": np.sin(6 * X[:, 0]) + X[:, 1] ** 2,
            "valid_y": 1,
        }
    )


def _linear() -> SingleTaskGPSurrogate:
    return SingleTaskGPSurrogate(inputs=INPUTS, outputs=OUTPUTS, kernel=LinearKernel())


def _rbf() -> SingleTaskGPSurrogate:
    return SingleTaskGPSurrogate(inputs=INPUTS, outputs=OUTPUTS)


def test_options_are_the_36_combinations():
    options = SingleTaskGPSurrogate.options(INPUTS, OUTPUTS)

    assert len(options) == 36
    assert len({o.model_dump_json() for o in options}) == 36
    # every option has components of its own
    assert len({id(o.kernel) for o in options}) == 36
    assert len({id(o.likelihood) for o in options}) == 36


def test_options_contain_a_scaled_matern_with_hvarfner_priors():
    expected_kernel = ScaleKernel(
        base_kernel=MaternKernel(
            nu=2.5, ard=True, lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR()
        ),
        outputscale_prior=THREESIX_SCALE_PRIOR(),
    )

    matches = [
        o
        for o in SingleTaskGPSurrogate.options(INPUTS, OUTPUTS)
        if o.kernel == expected_kernel
    ]

    assert len(matches) == 1
    assert matches[0].likelihood.noise_prior == HVARFNER_NOISE_PRIOR()


def test_selection_chooses_the_better_candidate():
    surrogate = surrogates.map(
        SelectionSurrogate(
            inputs=INPUTS,
            outputs=OUTPUTS,
            candidates=[_linear(), _rbf()],
            random_state=0,
        )
    )
    experiments = _experiments(15)

    surrogate.fit(experiments)

    # the response is not linear, so the RBF kernel predicts it better
    assert surrogate.selected == 1
    assert list(surrogate.scores.index) == [0, 1]
    assert surrogate.scores.loc[1, "MAE"] < surrogate.scores.loc[0, "MAE"]
    assert surrogate.predict(experiments).shape == (15, 2)


def test_selection_prefers_the_earlier_of_equal_candidates():
    surrogate = surrogates.map(
        SelectionSurrogate(
            inputs=INPUTS, outputs=OUTPUTS, candidates=[_rbf(), _rbf()], random_state=0
        )
    )

    surrogate.fit(_experiments(10))

    assert surrogate.selected == 0


def test_selection_dump_restores_the_chosen_candidate():
    candidates = [_linear(), _rbf()]
    surrogate = surrogates.map(
        SelectionSurrogate(
            inputs=INPUTS, outputs=OUTPUTS, candidates=candidates, random_state=0
        )
    )
    experiments = _experiments(10)
    surrogate.fit(experiments)

    restored = surrogates.map(
        SelectionSurrogate(
            inputs=INPUTS,
            outputs=OUTPUTS,
            candidates=candidates,
            dump=surrogate.dumps(),
        )
    )

    assert restored.selected == surrogate.selected
    pd.testing.assert_frame_equal(
        restored.predict(experiments), surrogate.predict(experiments)
    )
