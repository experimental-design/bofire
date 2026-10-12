import numpy as np
import pandas as pd
import pytest

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import (
    AdditiveMapSaasKernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    ScaleKernel,
)
from bofire.data_models.likelihoods.api import GaussianLikelihood
from bofire.data_models.means.api import ConstantMean
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    GreaterThan,
    NormalPrior,
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


_THREESIX_LIKELIHOOD = GaussianLikelihood(
    noise_prior=THREESIX_NOISE_PRIOR(), noise_constraint=GreaterThan(lower_bound=1e-4)
)


@pytest.mark.parametrize(
    "surrogate, kernel, mean, likelihood",
    [
        (
            SingleTaskGPSurrogate.linear(INPUTS, OUTPUTS),
            LinearKernel(),
            ConstantMean(),
            _THREESIX_LIKELIHOOD,
        ),
        (
            SingleTaskGPSurrogate.polynomial(INPUTS, OUTPUTS, power=3),
            PolynomialKernel(power=3),
            ConstantMean(),
            _THREESIX_LIKELIHOOD,
        ),
        (
            SingleTaskGPSurrogate.additive_map_saas(INPUTS, OUTPUTS, n_taus=2),
            AdditiveMapSaasKernel(n_taus=2),
            ConstantMean(prior=NormalPrior(loc=0, scale=1), bounds=(-10, 10)),
            GaussianLikelihood(),
        ),
    ],
)
def test_factories_build_their_components(surrogate, kernel, mean, likelihood):
    assert surrogate.kernel == kernel
    assert surrogate.mean == mean
    assert surrogate.likelihood == likelihood


def test_selection_chooses_the_better_option():
    surrogate = surrogates.map(
        SelectionSurrogate(
            inputs=INPUTS,
            outputs=OUTPUTS,
            options=[_linear(), _rbf()],
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


def test_selection_prefers_the_earlier_of_equal_options():
    surrogate = surrogates.map(
        SelectionSurrogate(
            inputs=INPUTS, outputs=OUTPUTS, options=[_rbf(), _rbf()], random_state=0
        )
    )

    surrogate.fit(_experiments(10))

    assert surrogate.selected == 0


def test_selection_dump_restores_the_chosen_option():
    options = [_linear(), _rbf()]
    surrogate = surrogates.map(
        SelectionSurrogate(
            inputs=INPUTS, outputs=OUTPUTS, options=options, random_state=0
        )
    )
    experiments = _experiments(10)
    surrogate.fit(experiments)

    restored = surrogates.map(
        SelectionSurrogate(
            inputs=INPUTS,
            outputs=OUTPUTS,
            options=options,
            dump=surrogate.dumps(),
        )
    )

    assert restored.selected == surrogate.selected
    pd.testing.assert_frame_equal(restored.scores, surrogate.scores)
    pd.testing.assert_frame_equal(
        restored.predict(experiments), surrogate.predict(experiments)
    )
