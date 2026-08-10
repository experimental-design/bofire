import importlib

import pytest

from bofire.data_models.descriptor_generators.api import (
    Fingerprints,
    MordredDescriptors,
)
from bofire.data_models.domain.api import Inputs
from bofire.data_models.features.api import (
    ContinuousInput,
    SumFeature,
    WeightedSumFeature,
)


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


def test_engineered_feature_validation():
    sum_feature = SumFeature(key="sum1", features=["feat1", "feat2", "feat3"])
    inputs = Inputs(
        features=[
            ContinuousInput(key="feat1", bounds=(0, 1)),
            ContinuousInput(key="feat2", bounds=(0, 1)),
        ]
    )
    with pytest.raises(
        ValueError, match="The following features are missing in inputs:"
    ):
        sum_feature.validate_features(inputs)


def test_weighted_sum_feature_validation():
    weighted_sum_feature = WeightedSumFeature(
        key="w_sum1", features=["feat1", "feat2"], columns=["desc1", "desc2"]
    )
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="feat1", bounds=(0, 1), descriptors={"desc1": [0.5], "desc2": [0.5]}
            ),
            ContinuousInput(key="feat2", bounds=(0, 1)),
        ]
    )
    with pytest.raises(
        ValueError, match="feat2: descriptor columns .* are not available"
    ):
        weighted_sum_feature.validate_features(inputs)

    inputs = Inputs(
        features=[
            ContinuousInput(
                key="feat1", bounds=(0, 1), descriptors={"desc1": [0.5], "desc2": [0.5]}
            ),
            ContinuousInput(
                key="feat2", bounds=(0, 1), descriptors={"desc1": [0.5], "desc3": [0.5]}
            ),
        ]
    )
    with pytest.raises(
        ValueError, match="feat2: descriptor columns .* are not available"
    ):
        weighted_sum_feature.validate_features(inputs)


def test_weighted_mean_feature_validation():
    weighted_mean_feature = WeightedSumFeature(
        key="w_mean1",
        features=["feat1", "feat2"],
        columns=["desc1", "desc2"],
        normalize=True,
    )
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="feat1", bounds=(0, 1), descriptors={"desc1": [0.5], "desc2": [0.5]}
            ),
            ContinuousInput(key="feat2", bounds=(0, 1)),
        ]
    )
    with pytest.raises(
        ValueError, match="feat2: descriptor columns .* are not available"
    ):
        weighted_mean_feature.validate_features(inputs)

    inputs = Inputs(
        features=[
            ContinuousInput(
                key="feat1", bounds=(0, 1), descriptors={"desc1": [0.5], "desc2": [0.5]}
            ),
            ContinuousInput(
                key="feat2", bounds=(0, 1), descriptors={"desc1": [0.5], "desc3": [0.5]}
            ),
        ]
    )
    with pytest.raises(
        ValueError, match="feat2: descriptor columns .* are not available"
    ):
        weighted_mean_feature.validate_features(inputs)


def test_molecular_weighted_sum_feature_validation():
    mol_feature = WeightedSumFeature(
        key="mw_sum1",
        features=["m1", "m2"],
        columns=[],
        generators=[MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])],
    )
    inputs = Inputs(
        features=[
            ContinuousInput(key="m1", bounds=(0, 1), structure=["C"]),
            ContinuousInput(key="m2", bounds=(0, 1)),
        ]
    )
    with pytest.raises(ValueError, match="m2: has no .structure. column"):
        mol_feature.validate_features(inputs)

    inputs = Inputs(
        features=[
            ContinuousInput(key="m1", bounds=(0, 1), structure=["C"]),
            ContinuousInput(key="m2", bounds=(0, 1), structure=["CC"]),
        ]
    )
    mol_feature.validate_features(inputs)


def test_molecular_weighted_mean_feature_validation():
    mol_feature = WeightedSumFeature(
        key="mw_mean1",
        features=["m1", "m2"],
        columns=[],
        generators=[MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])],
        normalize=True,
    )
    inputs = Inputs(
        features=[
            ContinuousInput(key="m1", bounds=(0, 1), structure=["C"]),
            ContinuousInput(key="m2", bounds=(0, 1)),
        ]
    )
    with pytest.raises(ValueError, match="m2: has no .structure. column"):
        mol_feature.validate_features(inputs)

    inputs = Inputs(
        features=[
            ContinuousInput(key="m1", bounds=(0, 1), structure=["C"]),
            ContinuousInput(key="m2", bounds=(0, 1), structure=["CC"]),
        ]
    )
    mol_feature.validate_features(inputs)


def _molecular_inputs() -> Inputs:
    return Inputs(
        features=[
            ContinuousInput(key="m1", bounds=(0, 1), structure=["C"]),
            ContinuousInput(key="m2", bounds=(0, 1), structure=["CC"]),
        ]
    )


def test_validate_features_does_not_generate_descriptor_values(monkeypatch):
    """Validation must stay free of descriptor generation.

    Generating needs rdkit, an optional extra, and `bofire.data_models` has to stay
    usable without it (the bare-install CI job asserts this). The width is resolved
    later, against the inputs, by `get_names`.
    """

    def boom(self, values):
        raise AssertionError("descriptor values generated during validation")

    monkeypatch.setattr(Fingerprints, "get_descriptor_values", boom)

    for filtering in (False, True):
        feature = WeightedSumFeature(
            key="w",
            features=["m1", "m2"],
            columns=[],
            generators=[Fingerprints(n_bits=8)],
            filter_descriptors=filtering,
        )
        feature.validate_features(_molecular_inputs())


# unlike the test above this one has to actually build the matrix, so it needs rdkit
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize("filtering", [False, True])
def test_get_names_width_matches_mapped_matrix(filtering):
    """`get_names` must agree with the matrix the mapper appends, filtered or not."""
    inputs = _molecular_inputs()
    feature = WeightedSumFeature(
        key="w",
        features=["m1", "m2"],
        columns=[],
        generators=[Fingerprints(n_bits=8)],
        filter_descriptors=filtering,
        correlation_cutoff=1.0,
    )
    components = [inputs.get_by_key(k) for k in feature.features]
    assert (
        len(feature.get_names(inputs)) == feature.component_table(components).shape[1]
    )
