"""What each kernel declares it can act on, and how that resolves against a domain."""

import pytest

from bofire.data_models.descriptor_generators.api import (
    Fingerprints,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.encodings.api import (
    DescriptorEncoding,
    OneHotEncoding,
    OrdinalEncoding,
)
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalTaskInput,
    ContinuousInput,
    ContinuousTaskInput,
    DiscreteInput,
    SumFeature,
)
from bofire.data_models.kernels.api import (
    HammingDistanceKernel,
    RBFKernel,
    TanimotoKernel,
    WassersteinKernel,
)
from bofire.data_models.kernels.fidelity import DownsamplingKernel


CAT = CategoricalInput(key="cat", categories=["a", "b"])
CONT = ContinuousInput(key="cont", bounds=(0, 1))
DISC = DiscreteInput(key="disc", values=[1.0, 2.0])
CAT_TASK = CategoricalTaskInput(key="task", categories=["t1", "t2"])
CONT_TASK = ContinuousTaskInput(key="fidelity", bounds=(0, 1))
FINGERPRINTS = DescriptorEncoding(generators=[Fingerprints()])
ENGINEERED = SumFeature(key="sum", features=["cont", "disc"])


@pytest.mark.parametrize(
    "kernel, feat, encoding, expected",
    [
        # a continuous kernel measures distance, so it needs coordinates
        (RBFKernel, CONT, None, True),
        (RBFKernel, DISC, None, True),
        (RBFKernel, CAT, OneHotEncoding(), True),
        (RBFKernel, CAT, DescriptorEncoding(), True),
        (RBFKernel, CAT, OrdinalEncoding(), False),
        # a task index is not a position in the space
        (RBFKernel, CAT_TASK, OrdinalEncoding(), False),
        (RBFKernel, CONT_TASK, None, False),
        # a categorical kernel compares by identity, so it needs one code per category
        (HammingDistanceKernel, CAT, OrdinalEncoding(), True),
        (HammingDistanceKernel, CAT, OneHotEncoding(), False),
        (HammingDistanceKernel, CAT, DescriptorEncoding(), False),
        (HammingDistanceKernel, CONT, None, False),
        # a molecular kernel needs the encoding to generate structural features
        (TanimotoKernel, CAT, FINGERPRINTS, True),
        (TanimotoKernel, CAT, DescriptorEncoding(generators=[Fragments()]), True),
        (
            TanimotoKernel,
            CAT,
            DescriptorEncoding(generators=[MordredDescriptors()]),
            False,
        ),
        (TanimotoKernel, CAT, DescriptorEncoding(), False),
        (TanimotoKernel, CONT, None, False),
        # an engineered feature is a number: only a continuous kernel can take it
        (RBFKernel, ENGINEERED, None, True),
        (HammingDistanceKernel, ENGINEERED, None, False),
        (TanimotoKernel, ENGINEERED, None, False),
        (DownsamplingKernel, ENGINEERED, None, False),
        # a fidelity kernel encodes a continuous task
        (DownsamplingKernel, CONT_TASK, None, True),
        (DownsamplingKernel, CONT, None, False),
        (DownsamplingKernel, CAT, OrdinalEncoding(), False),
    ],
)
def test_can_consume(kernel, feat, encoding, expected):
    assert kernel.can_consume(feat, encoding) is expected


def test_kernels_without_a_declared_rule_stay_permissive():
    """Shape kernels keep today's behaviour until their semantics are pinned down."""
    assert WassersteinKernel.can_consume(CONT, None) is True
    assert WassersteinKernel.can_consume(CAT, OrdinalEncoding()) is True


def test_accepted_encodings_orders_and_counts():
    """The first entry is what a kernel wants; the count is how particular it is."""
    candidates = [OrdinalEncoding(), OneHotEncoding(), DescriptorEncoding()]

    hamming = HammingDistanceKernel.accepted_encodings(CAT, candidates)
    rbf = RBFKernel.accepted_encodings(CAT, candidates)

    assert hamming == (OrdinalEncoding(),)
    assert rbf == (OneHotEncoding(), DescriptorEncoding())
    # the categorical kernel has the stronger claim on a categorical feature
    assert len(hamming) < len(rbf)


def test_accepted_encodings_is_empty_when_nothing_fits():
    assert HammingDistanceKernel.accepted_encodings(CONT, [OneHotEncoding()]) == ()


def test_resolve_features_filters_by_kind_when_unset():
    inputs = Inputs(features=[CONT, CAT])
    encodings = {"cat": OrdinalEncoding()}

    assert RBFKernel().resolve_features(inputs, encodings) == ["cont"]
    assert HammingDistanceKernel().resolve_features(inputs, encodings) == ["cat"]


def test_resolve_features_returns_an_explicit_list_unchanged():
    inputs = Inputs(features=[CONT, CAT])
    encodings = {"cat": OneHotEncoding()}

    assert RBFKernel(features=["cat"]).resolve_features(inputs, encodings) == ["cat"]


def test_resolve_features_can_be_empty():
    """A continuous kernel over a purely categorical domain selects nothing.

    The mixed GP relies on this to degenerate to a purely categorical model, so it is
    not an error at the kernel level.
    """
    inputs = Inputs(features=[CAT])

    assert RBFKernel().resolve_features(inputs, {"cat": OrdinalEncoding()}) == []


def test_engineered_features_are_filtered_like_any_other():
    """They are numeric columns, so only a kernel that takes numbers gets them."""
    inputs = Inputs(features=[CONT, DISC, CAT])
    engineered = EngineeredFeatures(features=[ENGINEERED])
    encodings = {"cat": OrdinalEncoding()}

    assert RBFKernel().resolve_features(inputs, encodings, engineered) == [
        "cont",
        "disc",
        "sum",
    ]
    # a categorical kernel must not pick up the sum
    assert HammingDistanceKernel().resolve_features(inputs, encodings, engineered) == [
        "cat"
    ]


def test_validate_inputs_rejects_an_unknown_key():
    inputs = Inputs(features=[CONT, CAT])

    with pytest.raises(ValueError, match="neither inputs nor engineered features"):
        RBFKernel(features=["nope"]).validate_inputs(inputs, {})


def test_validate_inputs_accepts_a_named_engineered_feature():
    inputs = Inputs(features=[CONT, DISC])
    engineered = EngineeredFeatures(features=[ENGINEERED])

    RBFKernel(features=["sum"]).validate_inputs(inputs, {}, engineered)


def test_validate_inputs_rejects_an_explicitly_named_feature_it_cannot_consume():
    inputs = Inputs(features=[CONT, CAT])
    encodings = {"cat": OrdinalEncoding()}

    with pytest.raises(ValueError, match=r"RBFKernel cannot act on \['cat'\]"):
        RBFKernel(features=["cat"]).validate_inputs(inputs, encodings)


def test_validate_inputs_is_silent_when_features_are_unset():
    """An unset selection is filtered, not rejected."""
    inputs = Inputs(features=[CONT, CAT])

    RBFKernel().validate_inputs(inputs, {"cat": OrdinalEncoding()})
