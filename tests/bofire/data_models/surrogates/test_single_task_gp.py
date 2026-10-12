import pytest

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import (
    AdditiveMapSaasKernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    ScaleKernel,
)
from bofire.data_models.likelihoods.api import HVARFNER_LIKELIHOOD, THREESIX_LIKELIHOOD
from bofire.data_models.means.api import ConstantMean
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
    NormalPrior,
)
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate


INPUTS = Inputs(
    features=[ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(2)]
)
OUTPUTS = Outputs(features=[ContinuousOutput(key="y")])


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


@pytest.mark.parametrize(
    "surrogate, kernel, mean, likelihood",
    [
        (
            SingleTaskGPSurrogate.linear(INPUTS, OUTPUTS),
            LinearKernel(),
            ConstantMean(),
            THREESIX_LIKELIHOOD(),
        ),
        (
            SingleTaskGPSurrogate.polynomial(INPUTS, OUTPUTS, power=3),
            PolynomialKernel(power=3),
            ConstantMean(),
            THREESIX_LIKELIHOOD(),
        ),
        (
            SingleTaskGPSurrogate.additive_map_saas(INPUTS, OUTPUTS, n_taus=2),
            AdditiveMapSaasKernel(n_taus=2),
            ConstantMean(prior=NormalPrior(loc=0, scale=1), bounds=(-10, 10)),
            HVARFNER_LIKELIHOOD(),
        ),
    ],
)
def test_factories_build_their_components(surrogate, kernel, mean, likelihood):
    assert surrogate.kernel == kernel
    assert surrogate.mean == mean
    assert surrogate.likelihood == likelihood
