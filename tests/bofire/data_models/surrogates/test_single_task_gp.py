import pytest

from bofire.data_models.descriptor_generators.api import Fingerprints
from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.encodings.api import DescriptorEncoding, OrdinalEncoding
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    SumFeature,
)
from bofire.data_models.features.descriptors import Descriptors
from bofire.data_models.kernels.api import (
    AdditiveMapSaasKernel,
    HammingDistanceKernel,
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
)
from bofire.data_models.likelihoods.api import (
    HVARFNER_LIKELIHOOD,
    THREESIX_LIKELIHOOD,
    SparseOutlierGaussianLikelihood,
)
from bofire.data_models.means.api import ConstantMean
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    ROBUSTGP_LENGTHSCALE_CONSTRAINT,
    THREESIX_SCALE_PRIOR,
    NonTransformedInterval,
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
            ConstantMean(
                prior=NormalPrior(loc=0, scale=1),
                constraint=NonTransformedInterval(lower_bound=-10, upper_bound=10),
            ),
            HVARFNER_LIKELIHOOD(),
        ),
        (
            SingleTaskGPSurrogate.robust(INPUTS, OUTPUTS),
            RBFKernel(
                ard=True,
                lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(),
            ),
            ConstantMean(),
            SparseOutlierGaussianLikelihood(noise_prior=HVARFNER_NOISE_PRIOR()),
        ),
    ],
)
def test_factories_build_their_components(surrogate, kernel, mean, likelihood):
    assert surrogate.kernel == kernel
    assert surrogate.mean == mean
    assert surrogate.likelihood == likelihood


@pytest.mark.parametrize(
    "factory",
    [
        SingleTaskGPSurrogate.linear,
        SingleTaskGPSurrogate.polynomial,
        SingleTaskGPSurrogate.additive_map_saas,
        SingleTaskGPSurrogate.robust,
        lambda *args, **kwargs: SingleTaskGPSurrogate.options(*args, **kwargs)[0],
    ],
)
def test_factories_take_engineered_features(factory):
    engineered = EngineeredFeatures(
        features=[SumFeature(key="s", features=["x0", "x1"])]
    )

    surrogate = factory(INPUTS, OUTPUTS, engineered_features=engineered)

    assert surrogate.engineered_features == engineered


MIXED_INPUTS = Inputs(
    features=[
        ContinuousInput(key="x0", bounds=(0, 1)),
        ContinuousInput(key="x1", bounds=(0, 1)),
        CategoricalInput(key="c", categories=["a", "b"]),
        CategoricalInput(
            key="d",
            categories=["a", "b"],
            descriptors=Descriptors(columns={"size": [1.0, 2.0]}),
        ),
    ]
)


def test_mixed_routes_plain_categoricals_to_the_hamming_kernel():
    engineered = EngineeredFeatures(
        features=[SumFeature(key="s", features=["x0", "x1"])]
    )

    surrogate = SingleTaskGPSurrogate.mixed(
        MIXED_INPUTS, OUTPUTS, engineered_features=engineered
    )

    sum_part, product_part = surrogate.kernel.kernels
    continuous, scaled_categorical = sum_part.base_kernel.kernels
    # descriptor-carrying categoricals and engineered features are continuous columns
    assert continuous.features == ["x0", "x1", "d", "s"]
    assert scaled_categorical.base_kernel.features == ["c"]
    assert [k.features for k in product_part.base_kernel.kernels] == [
        ["x0", "x1", "d", "s"],
        ["c"],
    ]
    # each occurrence has hyperparameters of its own
    assert continuous is not product_part.base_kernel.kernels[0]
    assert isinstance(surrogate.categorical_encodings["c"], OrdinalEncoding)
    assert not isinstance(surrogate.categorical_encodings["d"], OrdinalEncoding)


def test_mixed_without_continuous_inputs_is_a_scaled_hamming_kernel():
    inputs = Inputs(features=[CategoricalInput(key="c", categories=["a", "b"])])

    kernel = SingleTaskGPSurrogate.mixed(inputs, OUTPUTS).kernel

    assert isinstance(kernel, ScaleKernel)
    assert kernel.base_kernel == HammingDistanceKernel(ard=True, features=["c"])


@pytest.mark.parametrize(
    "inputs, engineered, message",
    [
        (
            Inputs(features=[ContinuousInput(key="x0", bounds=(0, 1))]),
            None,
            "at least one categorical input without descriptor data",
        ),
        (
            MIXED_INPUTS,
            EngineeredFeatures(
                features=[
                    SumFeature(key="s", features=["x0", "x1"], keep_features=False)
                ]
            ),
            "keep the features it is computed from",
        ),
    ],
)
def test_mixed_rejects(inputs, engineered, message):
    with pytest.raises(ValueError, match=message):
        SingleTaskGPSurrogate.mixed(inputs, OUTPUTS, engineered_features=engineered)


MOLECULES = CategoricalInput(
    key="mol",
    categories=["C", "CC"],
    descriptors=Descriptors(structure=["C", "CC"]),
)


def test_tanimoto_precomputes_the_similarities_of_the_molecular_input():
    surrogate = SingleTaskGPSurrogate.tanimoto(Inputs(features=[MOLECULES]), OUTPUTS)

    assert surrogate.kernel == ScaleKernel(
        base_kernel=TanimotoKernel(
            features=["mol"],
            fingerprints=DescriptorEncoding(columns=[], generators=[Fingerprints()]),
        ),
        outputscale_prior=THREESIX_SCALE_PRIOR(),
    )
    assert surrogate.categorical_encodings == {"mol": OrdinalEncoding()}
    assert surrogate.likelihood == THREESIX_LIKELIHOOD()


@pytest.mark.parametrize(
    "inputs",
    [
        Inputs(features=[CategoricalInput(key="c", categories=["a", "b"])]),
        Inputs(features=[MOLECULES, ContinuousInput(key="x", bounds=(0, 1))]),
    ],
)
def test_tanimoto_rejects(inputs):
    with pytest.raises(ValueError, match="exactly one input"):
        SingleTaskGPSurrogate.tanimoto(inputs, OUTPUTS)
