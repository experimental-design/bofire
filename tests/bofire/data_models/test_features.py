import importlib
import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pydantic.error_wrappers import ValidationError

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.domain.api import Features, Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
    MolecularInput,
    Output,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.objectives.api import MinimizeObjective, Objective
from bofire.data_models.surrogates.api import ScalerEnum

objective = MinimizeObjective(w=1)

RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


# test features container
if1 = specs.features.valid(ContinuousInput).obj(key="if1")
if2 = specs.features.valid(ContinuousInput).obj(key="if2")
if3 = specs.features.valid(ContinuousInput).obj(key="if3", bounds=(3, 3))
if4 = specs.features.valid(CategoricalInput).obj(
    key="if4", categories=["a", "b"], allowed=[True, False]
)
if5 = specs.features.valid(DiscreteInput).obj(key="if5")
if7 = specs.features.valid(CategoricalInput).obj(
    key="if7",
    categories=["c", "d", "e"],
    allowed=[True, False, False],
)


of1 = specs.features.valid(ContinuousOutput).obj(key="of1")
of2 = specs.features.valid(ContinuousOutput).obj(key="of2")
of3 = specs.features.valid(ContinuousOutput).obj(key="of3", objective=None)

inputs = Inputs(features=[if1, if2])
outputs = Outputs(features=[of1, of2])
features = Features(features=[if1, if2, of1, of2])


@pytest.mark.parametrize(
    "FeatureContainer, features",
    [
        (Features, ["s"]),
        (Features, [specs.features.valid(ContinuousInput).obj(), 5]),
        (Inputs, ["s"]),
        (Inputs, [specs.features.valid(ContinuousInput).obj(), 5]),
        (
            Inputs,
            [
                specs.features.valid(ContinuousInput).obj(),
                specs.features.valid(ContinuousOutput).obj(),
            ],
        ),
        (Outputs, ["s"]),
        (Outputs, [specs.features.valid(ContinuousOutput).obj(), 5]),
        (
            Outputs,
            [
                specs.features.valid(ContinuousOutput).obj(),
                specs.features.valid(ContinuousInput).obj(),
            ],
        ),
    ],
)
def test_features_invalid_feature(FeatureContainer, features):
    with pytest.raises((ValueError, TypeError, KeyError, ValidationError)):
        FeatureContainer(features=features)


@pytest.mark.parametrize(
    "features1, features2, expected_type",
    [
        [inputs, inputs, Inputs],
        [outputs, outputs, Outputs],
        [inputs, outputs, Features],
        [outputs, inputs, Features],
        [features, outputs, Features],
        [features, inputs, Features],
        [outputs, features, Features],
        [inputs, features, Features],
    ],
)
def test_features_plus(features1, features2, expected_type):
    returned = features1 + features2
    assert type(returned) == expected_type
    assert len(returned) == (len(features1) + len(features2))


@pytest.mark.parametrize(
    "features, FeatureType, exact, expected",
    [
        (features, Feature, False, [if1, if2, of1, of2]),
        (features, Output, False, [of1, of2]),
        (inputs, ContinuousInput, False, [if1, if2]),
        (outputs, ContinuousOutput, False, [of1, of2]),
    ],
)
def test_constraints_get(features, FeatureType, exact, expected):
    returned = features.get(FeatureType, exact=exact)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])
    assert type(returned) == type(features)


@pytest.mark.parametrize(
    "features, FeatureType, exact, expected",
    [
        (features, Feature, False, ["if1", "if2", "of1", "of2"]),
        (features, Output, False, ["of1", "of2"]),
        (inputs, ContinuousInput, False, ["if1", "if2"]),
        (outputs, ContinuousOutput, False, ["of1", "of2"]),
    ],
)
def test_features_get_keys(features, FeatureType, exact, expected):
    assert features.get_keys(FeatureType, exact=exact) == expected


@pytest.mark.parametrize(
    "features, key, expected",
    [
        (features, "if1", if1),
        (outputs, "of1", of1),
        (inputs, "if1", if1),
    ],
)
def test_features_get_by_key(features, key, expected):
    returned = features.get_by_key(key)
    assert returned.key == expected.key
    assert id(returned) == id(expected)


def test_features_get_by_keys():
    keys = ["of2", "if1"]
    feats = features.get_by_keys(keys)
    assert feats[0].key == "if1"
    assert feats[1].key == "of2"


@pytest.mark.parametrize(
    "features, key",
    [
        (features, "if133"),
        (outputs, "of3331"),
        (inputs, "if1333333"),
    ],
)
def test_features_get_by_key_invalid(features, key):
    with pytest.raises(KeyError):
        features.get_by_key(key)


@pytest.mark.parametrize(
    "features, expected",
    [
        (Inputs(features=[if1, if2]), []),
        (Inputs(features=[if1, if2, if3, if4, if5]), [if3, if4]),
    ],
)
def test_inputs_get_fixed(features, expected):
    returned = features.get_fixed()
    assert isinstance(features, Inputs)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])


@pytest.mark.parametrize(
    "features, expected",
    [
        (Inputs(features=[if1, if2]), [if1, if2]),
        (Inputs(features=[if1, if2, if3, if4, if5]), [if1, if2, if5]),
    ],
)
def test_inputs_get_free(features, expected):
    returned = features.get_free()
    assert isinstance(features, Inputs)
    assert returned.features == expected
    for i in range(len(expected)):
        assert id(expected[i]) == id(returned[i])


@pytest.mark.parametrize(
    "features, num_samples, method",
    [
        (features, num_samples, method)
        for features in [
            inputs,
            Inputs(features=[if1, if2, if3, if4, if5, if7]),
        ]
        for num_samples in [1, 2, 1024]
        for method in ["UNIFORM", "SOBOL", "LHS"]
    ],
)
def test_inputs_sample(features: Inputs, num_samples, method):
    samples = features.sample(num_samples, method=method)
    assert samples.shape == (num_samples, len(features))
    assert list(samples.columns) == features.get_keys()


@pytest.mark.parametrize(
    "specs",
    [
        ({"x4": CategoricalEncodingEnum.ONE_HOT}),
        ({"x1": CategoricalEncodingEnum.ONE_HOT}),
        ({"x2": ScalerEnum.NORMALIZE}),
        ({"x2": CategoricalEncodingEnum.DESCRIPTOR}),
        ({"x1": Fingerprints()}),
        ({"x2": Fragments()}),
        ({"x3": FingerprintsFragments()}),
        ({"x3": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])}),
    ],
)
def test_inputs_validate_transform_specs_invalid(specs):
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4]],
            ),
        ]
    )
    with pytest.raises(ValueError):
        inps._validate_transform_specs(specs)


@pytest.mark.parametrize(
    "specs",
    [
        ({"x2": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.DESCRIPTOR}),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
            }
        ),
    ],
)
def test_inputs_validate_transform_valid(specs):
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4]],
            ),
        ]
    )
    inps._validate_transform_specs(specs)


@pytest.mark.parametrize(
    "specs",
    [
        # ({"x2": CategoricalEncodingEnum.ONE_HOT}),
        # ({"x3": CategoricalEncodingEnum.DESCRIPTOR}),
        ({"x4": CategoricalEncodingEnum.ONE_HOT}),
        ({"x4": ScalerEnum.NORMALIZE}),
        ({"x4": CategoricalEncodingEnum.DESCRIPTOR}),
        # (
        #    {
        #        "x2": CategoricalEncodingEnum.ONE_HOT,
        #        "x3": CategoricalEncodingEnum.DESCRIPTOR,
        #    }
        # ),
    ],
)
def test_inputs_validate_transform_specs_molecular_input_invalid(specs):
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4]],
            ),
            MolecularInput(key="x4"),
        ]
    )
    with pytest.raises(ValueError):
        inps._validate_transform_specs(specs)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "specs",
    [
        ({"x4": Fingerprints()}),
        ({"x4": Fragments()}),
        ({"x4": FingerprintsFragments()}),
        ({"x4": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])}),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x4": Fingerprints(),
            }
        ),
        (
            {
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": Fingerprints(),
            }
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": Fingerprints(),
            }
        ),
    ],
)
def test_inputs_validate_transform_specs_molecular_input_valid(specs):
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4]],
            ),
            MolecularInput(key="x4"),
        ]
    )
    inps._validate_transform_specs(specs)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "specs, expected_features2idx, expected_features2names",
    [
        (
            {"x2": CategoricalEncodingEnum.ONE_HOT, "x4": Fingerprints(n_bits=2048)},
            {
                "x1": (2048,),
                "x2": (2050, 2051, 2052),
                "x3": (2049,),
                "x4": tuple(range(2048)),
            },
            {
                "x1": ("x1",),
                "x2": ("x2_apple", "x2_banana", "x2_orange"),
                "x3": ("x3",),
                "x4": tuple(f"x4_fingerprint_{i}" for i in range(2048)),
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.DUMMY,
                "x4": Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            },
            {"x1": (2,), "x2": (4, 5), "x3": (3,), "x4": (0, 1)},
            {
                "x1": ("x1",),
                "x2": ("x2_banana", "x2_orange"),
                "x3": ("x3",),
                "x4": ("x4_fr_unbrch_alkane", "x4_fr_thiocyan"),
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ORDINAL,
                "x4": FingerprintsFragments(
                    n_bits=2048, fragments=["fr_unbrch_alkane", "fr_thiocyan"]
                ),
            },
            {
                "x1": (2050,),
                "x2": (2052,),
                "x3": (2051,),
                "x4": tuple(range(2048 + 2)),
            },
            {
                "x1": ("x1",),
                "x2": ("x2",),
                "x3": ("x3",),
                "x4": tuple(
                    [f"x4_fingerprint_{i}" for i in range(2048)]
                    + ["x4_fr_unbrch_alkane", "x4_fr_thiocyan"]
                ),
            },
        ),
        (
            {
                "x3": CategoricalEncodingEnum.ONE_HOT,
                "x4": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            {"x1": (2,), "x2": (7,), "x3": (3, 4, 5, 6), "x4": (0, 1)},
            {
                "x1": ("x1",),
                "x2": ("x2",),
                "x3": ("x3_apple", "x3_banana", "x3_orange", "x3_cherry"),
                "x4": ("x4_NssCH2", "x4_ATSC2d"),
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            {"x1": (2,), "x2": (5, 6, 7), "x3": (3, 4), "x4": (0, 1)},
            {
                "x1": ("x1",),
                "x2": ("x2_apple", "x2_banana", "x2_orange"),
                "x3": (
                    "x3_d1",
                    "x3_d2",
                ),
                "x4": ("x4_NssCH2", "x4_ATSC2d"),
            },
        ),
    ],
)
def test_inputs_get_transform_info(
    specs, expected_features2idx, expected_features2names
):
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana", "orange"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana", "orange", "cherry"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
            MolecularInput(key="x4"),
        ]
    )
    features2idx, features2names = inps._get_transform_info(specs)
    assert features2idx == expected_features2idx
    assert features2names == expected_features2names


@pytest.mark.parametrize(
    "specs",
    [
        ({"x2": CategoricalEncodingEnum.ONE_HOT}),
        ({"x2": CategoricalEncodingEnum.DUMMY}),
        ({"x2": CategoricalEncodingEnum.ORDINAL}),
        ({"x3": CategoricalEncodingEnum.ONE_HOT}),
        ({"x3": CategoricalEncodingEnum.DESCRIPTOR}),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
            }
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
            }
        ),
        (
            {
                "x2": CategoricalEncodingEnum.DUMMY,
                "x3": CategoricalEncodingEnum.ONE_HOT,
            }
        ),
    ],
)
def test_inputs_transform(specs):
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana", "orange"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana", "orange", "cherry"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
        ]
    )
    samples = inps.sample(n=100)
    samples = samples.sample(40)
    transformed = inps.transform(experiments=samples, specs=specs)
    untransformed = inps.inverse_transform(experiments=transformed, specs=specs)
    assert_frame_equal(samples, untransformed)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_input_reverse_transform_molecular():
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana", "orange"]),
            CategoricalMolecularInput(
                key="x3",
                categories=[
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "c1ccccc1",
                    "[CH3][CH2][OH]",
                    "N[C@](C)(F)C(=O)O",
                ],
            ),
        ],
    )
    specs = {
        "x3": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
        "x2": CategoricalEncodingEnum.ONE_HOT,
    }
    samples = inps.sample(n=20)
    transformed = inps.transform(experiments=samples, specs=specs)
    untransformed = inps.inverse_transform(experiments=transformed, specs=specs)
    assert_frame_equal(samples, untransformed)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "specs, expected",
    [
        (
            {"x2": CategoricalEncodingEnum.ONE_HOT, "x4": Fingerprints(n_bits=32)},
            {
                "x4_fingerprint_0": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_1": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                "x4_fingerprint_2": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x4_fingerprint_3": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_4": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_5": {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_6": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x4_fingerprint_7": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                "x4_fingerprint_8": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_9": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_10": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_11": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_12": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_13": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_14": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_15": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_16": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0},
                "x4_fingerprint_17": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_18": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_19": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_20": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_21": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_22": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_23": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_24": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_25": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_26": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_27": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_28": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_29": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_30": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x4_fingerprint_31": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x1": {0: 0.1, 1: 0.3, 2: 0.5, 3: 1.0},
                "x3": {0: "banana", 1: "orange", 2: "apple", 3: "cherry"},
                "x2_apple": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x2_banana": {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "x2_orange": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.DUMMY,
                "x4": Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            },
            {
                "x4_fr_unbrch_alkane": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fr_thiocyan": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x1": {0: 0.1, 1: 0.3, 2: 0.5, 3: 1.0},
                "x3": {0: "banana", 1: "orange", 2: "apple", 3: "cherry"},
                "x2_banana": {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "x2_orange": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ORDINAL,
                "x4": FingerprintsFragments(
                    n_bits=32, fragments=["fr_unbrch_alkane", "fr_thiocyan"]
                ),
            },
            {
                "x4_fingerprint_0": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_1": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                "x4_fingerprint_2": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x4_fingerprint_3": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_4": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_5": {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_6": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x4_fingerprint_7": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                "x4_fingerprint_8": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_9": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_10": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_11": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_12": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_13": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_14": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_15": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_16": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0},
                "x4_fingerprint_17": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_18": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_19": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_20": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_21": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_22": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_23": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_24": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_25": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_26": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_27": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_28": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fingerprint_29": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "x4_fingerprint_30": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x4_fingerprint_31": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fr_unbrch_alkane": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x4_fr_thiocyan": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "x1": {0: 0.1, 1: 0.3, 2: 0.5, 3: 1.0},
                "x3": {0: "banana", 1: "orange", 2: "apple", 3: "cherry"},
                "x2": {0: 0, 1: 1, 2: 0, 3: 2},
            },
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            {
                "x4_NssCH2": {
                    0: 0.5963718820861676,
                    1: -1.5,
                    2: -0.28395061728395066,
                    3: -8.34319526627219,
                },
                "x4_ATSC2d": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x1": {0: 0.1, 1: 0.3, 2: 0.5, 3: 1.0},
                "x3_d1": {0: 3.0, 1: 5.0, 2: 1.0, 3: 7.0},
                "x3_d2": {0: 4.0, 1: 6.0, 2: 2.0, 3: 8.0},
                "x2_apple": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "x2_banana": {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "x2_orange": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
            },
        ),
    ],
)
def test_inputs_transform_molecular(specs, expected):
    experiments = [
        [0.1, "apple", "banana", "CC(=O)Oc1ccccc1C(=O)O", 88.0],
        [0.3, "banana", "orange", "c1ccccc1", 35.0],
        [0.5, "apple", "apple", "[CH3][CH2][OH]", 69.0],
        [1.0, "orange", "cherry", "N[C@](C)(F)C(=O)O", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x1", "x2", "x3", "x4", "y"])
    experiments["valid_y"] = 1
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana", "orange"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana", "orange", "cherry"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
            MolecularInput(key="x4"),
        ]
    )
    transformed = inps.transform(experiments=experiments, specs=specs)
    assert_frame_equal(transformed, pd.DataFrame.from_dict(expected))


if1 = specs.features.valid(ContinuousInput).obj(key="if1")
if2 = specs.features.valid(ContinuousInput).obj(key="if2", bounds=(3, 3))
if3 = specs.features.valid(CategoricalInput).obj(
    key="if3",
    categories=["c1", "c2", "c3"],
    allowed=[True, True, True],
)
if4 = specs.features.valid(CategoricalInput).obj(
    key="if4",
    categories=["c1", "c2", "c3"],
    allowed=[True, False, False],
)
if5 = specs.features.valid(CategoricalDescriptorInput).obj(
    key="if5",
    categories=["c1", "c2", "c3"],
    allowed=[True, False, False],
    descriptors=["d1", "d2"],
    values=[
        [1, 2],
        [3, 7],
        [5, 1],
    ],
)
if6 = specs.features.valid(CategoricalDescriptorInput).obj(
    key="if6",
    categories=["c1", "c2", "c3"],
    allowed=[True, False, False],
    descriptors=["d1", "d2"],
    values=[
        [1, 2],
        [3, 7],
        [5, 1],
    ],
)

of1 = specs.features.valid(ContinuousOutput).obj(key="of1")

inputs1 = Inputs(features=[if1, if3, if5])

inputs2 = Inputs(
    features=[
        if1,
        if2,
        if3,
        if4,
        if5,
        if6,
    ]
)


@pytest.mark.parametrize(
    "inputs, specs, expected_bounds",
    [
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 2, 0, 0, 0], [5.3, 1, 2, 1, 1, 1]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.DUMMY,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 2, 0, 0], [5.3, 1, 2, 1, 1]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.DUMMY,
                "if5": CategoricalEncodingEnum.DUMMY,
            },
            [[3, 0, 0, 0, 0], [5.3, 1, 1, 1, 1]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ONE_HOT,
            },
            [[3, 0, 0, 0, 0, 0, 0], [5.3, 1, 0, 0, 1, 1, 1]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [[3, 1, 2, 0], [5.3, 1, 2, 2]],
        ),
        (
            inputs1,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.ORDINAL,
            },
            [[3, 0, 0], [5.3, 2, 2]],
        ),
        # new domain
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [3, 3, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0],
                [
                    5.3,
                    3,
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                ],
            ],
        ),
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ONE_HOT,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ONE_HOT,
                "if6": CategoricalEncodingEnum.ONE_HOT,
            },
            [
                [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5.3, 3, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            ],
        ),
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.DESCRIPTOR,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [
                    3,
                    3,
                    1,
                    2,
                    1,
                    2,
                    0,
                    0,
                ],
                [5.3, 3, 1, 2, 1, 2, 2, 2],
            ],
        ),
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ORDINAL,
                "if5": CategoricalEncodingEnum.ORDINAL,
                "if6": CategoricalEncodingEnum.ORDINAL,
            },
            [[3, 3, 0, 0, 0, 0], [5.3, 3, 2, 2, 2, 2]],
        ),
        (
            inputs2,
            {
                "if3": CategoricalEncodingEnum.ORDINAL,
                "if4": CategoricalEncodingEnum.ONE_HOT,
                "if5": CategoricalEncodingEnum.ORDINAL,
                "if6": CategoricalEncodingEnum.DESCRIPTOR,
            },
            [
                [3.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [
                    5.3,
                    3.0,
                    2.0,
                    1.0,
                    2.0,
                    2.0,
                    1.0,
                    0.0,
                    0.0,
                ],
            ],
        ),
    ],
)
def test_inputs_get_bounds(inputs, specs, expected_bounds):
    lower, upper = inputs.get_bounds(specs=specs)
    assert np.allclose(expected_bounds[0], lower)
    assert np.allclose(expected_bounds[1], upper)


def test_inputs_get_bounds_fit():
    # at first the fix on the continuous ones is tested
    inputs = Inputs(features=[if1, if2])
    experiments = inputs.sample(100)
    experiments["if1"] += [random.uniform(-2, 2) for _ in range(100)]
    experiments["if2"] += [random.uniform(-2, 2) for _ in range(100)]
    opt_bounds = inputs.get_bounds(specs={})
    fit_bounds = inputs.get_bounds(specs={}, experiments=experiments)
    for i, key in enumerate(inputs.get_keys(ContinuousInput)):
        assert fit_bounds[0][i] < opt_bounds[0][i]
        assert fit_bounds[1][i] > opt_bounds[1][i]
        assert fit_bounds[0][i] == experiments[key].min()
        assert fit_bounds[1][i] == experiments[key].max()
    # next test the fix for the CategoricalDescriptor feature
    inputs = Inputs(
        features=[
            if1,
            if2,
            if3,
            if4,
            if5,
            if6,
        ]
    )
    experiments = inputs.sample(100)
    experiments["if4"] = [random.choice(if4.categories) for _ in range(100)]
    experiments["if6"] = [random.choice(if6.categories) for _ in range(100)]
    opt_bounds = inputs.get_bounds(
        specs={
            "if3": CategoricalEncodingEnum.ONE_HOT,
            "if4": CategoricalEncodingEnum.ONE_HOT,
            "if5": CategoricalEncodingEnum.DESCRIPTOR,
            "if6": CategoricalEncodingEnum.DESCRIPTOR,
        }
    )
    fit_bounds = inputs.get_bounds(
        {
            "if3": CategoricalEncodingEnum.ONE_HOT,
            "if4": CategoricalEncodingEnum.ONE_HOT,
            "if5": CategoricalEncodingEnum.DESCRIPTOR,
            "if6": CategoricalEncodingEnum.DESCRIPTOR,
        },
        experiments=experiments,
    )
    # check difference in descriptors
    assert opt_bounds[0][-8] == 1
    assert opt_bounds[1][-8] == 1
    assert opt_bounds[0][-7] == 2
    assert opt_bounds[1][-7] == 2
    assert fit_bounds[0][-8] == 1
    assert fit_bounds[0][-7] == 1
    assert fit_bounds[1][-8] == 5
    assert fit_bounds[1][-7] == 7
    # check difference in onehots
    assert opt_bounds[1][-1] == 0
    assert opt_bounds[1][-2] == 0
    assert fit_bounds[1][-1] == 1
    assert fit_bounds[1][-2] == 1


mixed_data = pd.DataFrame(
    columns=["of1", "of2", "of3"],
    index=range(5),
    data=np.random.uniform(size=(5, 3)),
)
mixed_data["of4"] = ["a", "a", "b", "b", "a"]


@pytest.mark.parametrize(
    "features, samples",
    [
        (
            outputs,
            pd.DataFrame(
                columns=["of1", "of2"],
                index=range(5),
                data=np.random.uniform(size=(5, 2)),
            ),
        ),
        (
            Outputs(features=[of1, of2, of3]),
            pd.DataFrame(
                columns=["of1", "of2", "of3"],
                index=range(5),
                data=np.random.uniform(size=(5, 3)),
            ),
        ),
        (
            Outputs(
                features=[
                    of1,
                    of2,
                    of3,
                    CategoricalOutput(
                        key="of4", categories=["a", "b"], objective=[1.0, 0.0]
                    ),
                ]
            ),
            mixed_data,
        ),
    ],
)
def test_outputs_call(features, samples):
    o = features(samples)
    assert o.shape == (
        len(samples),
        len(features.get_keys_by_objective(Objective))
        + len(features.get_keys(CategoricalOutput)),
    )
    assert list(o.columns) == [
        f"{key}_des"
        for key in features.get_keys_by_objective(Objective)
        + features.get_keys(CategoricalOutput)
    ]


def test_categorical_output():
    feature = CategoricalOutput(
        key="a", categories=["alpha", "beta", "gamma"], objective=[1.0, 0.0, 0.1]
    )

    assert feature.to_dict() == {"alpha": 1.0, "beta": 0.0, "gamma": 0.1}
    data = pd.Series(data=["alpha", "beta", "beta", "gamma"], name="a")
    assert_series_equal(feature(data), pd.Series(data=[1.0, 0.0, 0.0, 0.1], name="a"))
