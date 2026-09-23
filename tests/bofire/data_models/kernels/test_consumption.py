"""Whether a kernel can work on the features it is applied to."""

import pytest

from bofire.data_models.constraints.condition import NonZeroCondition
from bofire.data_models.descriptor_generators.api import Fingerprints
from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.encodings.api import OneHotEncoding, OrdinalEncoding
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousTaskInput,
    DiscreteInput,
    WeightedSumFeature,
)
from bofire.data_models.features.descriptors import Descriptors
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    HammingDistanceKernel,
    IndexKernel,
    PositiveIndexKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
    WedgeKernel,
)
from bofire.data_models.kernels.fidelity import DownsamplingKernel
from bofire.data_models.kernels.kernel import KernelInputs


def make_context(*features, encodings=None, engineered=(), offered=None):
    inputs = Inputs(features=list(features))
    engineered_features = EngineeredFeatures(features=list(engineered))
    keys = inputs.get_keys() + engineered_features.get_keys()
    return KernelInputs(
        inputs=inputs,
        encodings=encodings or {},
        engineered_features=engineered_features,
        offered=tuple(keys if offered is None else offered),
    )


CONT = ContinuousInput(key="cont", bounds=(0, 1))
DISC = DiscreteInput(key="disc", values=[1.0, 2.0])
CAT = CategoricalInput(key="cat", categories=["a", "b"])
CAT3 = CategoricalInput(key="cat3", categories=["a", "b", "c"])
FIDELITY = ContinuousTaskInput(key="fidelity", bounds=(0, 1))


@pytest.mark.parametrize(
    "kernel, context",
    [
        # modelling choices that are unusual but computable stay allowed
        pytest.param(
            RBFKernel(),
            make_context(CONT, CAT, encodings={"cat": OrdinalEncoding()}),
            id="continuous-kernel-on-ordinal-codes",
        ),
        pytest.param(
            HammingDistanceKernel(features=["cat"]),
            make_context(CONT, CAT, encodings={"cat": OneHotEncoding()}),
            id="hamming-on-one-hot",
        ),
        pytest.param(
            HammingDistanceKernel(features=["disc"]),
            make_context(CONT, DISC),
            id="hamming-on-discrete",
        ),
        pytest.param(
            TanimotoKernel(features=["blend"]),
            make_context(
                ContinuousInput(
                    key="a", bounds=(0, 1), descriptors=Descriptors(structure=["CCO"])
                ),
                ContinuousInput(
                    key="b", bounds=(0, 1), descriptors=Descriptors(structure=["CC"])
                ),
                engineered=[
                    WeightedSumFeature(
                        key="blend",
                        features=["a", "b"],
                        columns=[],
                        generators=[Fingerprints()],
                    )
                ],
            ),
            id="tanimoto-on-fingerprint-blend",
        ),
        pytest.param(
            IndexKernel(num_categories=2, features=["cat"]),
            make_context(CONT, CAT, encodings={"cat": OrdinalEncoding()}),
            id="index-on-one-ordinal-categorical",
        ),
        pytest.param(
            DownsamplingKernel(features=["fidelity"]),
            make_context(CONT, FIDELITY),
            id="downsampling-on-fidelity",
        ),
    ],
)
def test_validate_inputs_accepts(kernel, context):
    kernel.validate_inputs(context)


@pytest.mark.parametrize(
    "kernel, context, match",
    [
        pytest.param(
            HammingDistanceKernel(features=["cont"]),
            make_context(CONT, CAT, encodings={"cat": OrdinalEncoding()}),
            r"HammingDistanceKernel cannot work on \['cont'\]",
            id="categorical-kernel-on-continuous",
        ),
        pytest.param(
            RBFKernel(features=["nope"]),
            make_context(CONT),
            r"names \['nope'\], which are neither inputs nor engineered features",
            id="unknown-key",
        ),
        pytest.param(
            HammingDistanceKernel(),
            make_context(CONT, CAT, encodings={"cat": OrdinalEncoding()}),
            r"HammingDistanceKernel cannot work on \['cont'\]",
            id="unset-features-are-checked-not-filtered",
        ),
        pytest.param(
            IndexKernel(num_categories=2),
            make_context(
                CAT,
                CAT3,
                encodings={"cat": OrdinalEncoding(), "cat3": OrdinalEncoding()},
            ),
            "works on exactly one feature",
            id="index-on-two-features",
        ),
        pytest.param(
            PositiveIndexKernel(num_categories=2, features=["cat3"]),
            make_context(CAT3, encodings={"cat3": OrdinalEncoding()}),
            "num_categories=2, but 'cat3' has 3 categories",
            id="index-category-count-mismatch",
        ),
        pytest.param(
            IndexKernel(num_categories=2, features=["cat"]),
            make_context(CAT, encodings={"cat": OneHotEncoding()}),
            "needs an ordinal-encoded categorical",
            id="index-on-one-hot",
        ),
        pytest.param(
            DownsamplingKernel(features=["cont"]),
            make_context(CONT),
            r"DownsamplingKernel cannot work on \['cont'\]",
            id="downsampling-on-plain-continuous",
        ),
        pytest.param(
            WedgeKernel(
                base_kernel=RBFKernel(),
                conditions=[("cont", "missing", NonZeroCondition())],
            ),
            make_context(CONT),
            r"conditions on \['missing'\]",
            id="wedge-condition-on-unknown-key",
        ),
    ],
)
def test_validate_inputs_rejects(kernel, context, match):
    with pytest.raises(ValueError, match=match):
        kernel.validate_inputs(context)


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(
            ScaleKernel(base_kernel=HammingDistanceKernel(features=["cont"])),
            id="inside-scale",
        ),
        pytest.param(
            AdditiveKernel(
                kernels=[
                    RBFKernel(features=["cont"]),
                    ScaleKernel(base_kernel=HammingDistanceKernel(features=["cont"])),
                ]
            ),
            id="nested-in-additive",
        ),
        pytest.param(
            WedgeKernel(
                base_kernel=HammingDistanceKernel(features=["cont"]),
                conditions=[("cont", "cont", NonZeroCondition())],
            ),
            id="as-wedge-base-kernel",
        ),
    ],
)
def test_validate_inputs_reaches_into_composites(kernel):
    with pytest.raises(ValueError, match=r"HammingDistanceKernel cannot work"):
        kernel.validate_inputs(make_context(CONT))


def test_unset_features_select_what_is_offered():
    context = make_context(CONT, DISC, CAT, offered=["cont", "disc"])

    assert RBFKernel().selected_features(context) == ["cont", "disc"]
    assert RBFKernel(features=["cat"]).selected_features(context) == ["cat"]


def test_validate_inputs_does_not_modify_the_kernel():
    kernel = ScaleKernel(base_kernel=RBFKernel())
    before = kernel.model_dump()

    kernel.validate_inputs(make_context(CONT, CAT, encodings={"cat": OneHotEncoding()}))

    assert kernel.model_dump() == before
    assert kernel.base_kernel.features is None
