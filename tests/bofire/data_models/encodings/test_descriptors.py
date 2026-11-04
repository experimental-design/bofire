import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bofire.data_models.descriptor_generators.api import (
    Fingerprints,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.encodings.api import DescriptorEncoding
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.features.descriptors import Descriptors


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "[CH3][CH2][OH]",
    "N[C@](C)(F)C(=O)O",
]
VALID_SMILES = pd.Series(smiles)
VALID_SMILES.name = "molecule"


# --- from static descriptor columns ---


@pytest.mark.parametrize(
    "key, categories, samples_in, descriptors",
    [
        ("c", ["B", "A", "C"], ["A", "A", "C", "B"], ["d1", "d2"]),
        (
            "c_alpha",
            ["B_b", "_A_a", "C_c_"],
            ["_A_a", "_A_a", "C_c_", "B_b"],
            ["_d1_d", "d2_d_2_"],
        ),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
            [
                "__c_alpha_c___A_a",
                "__c_alpha_c___A_a",
                "__c_alpha_c__C_c_",
                "__c_alpha_c__B_b",
            ],
            ["__c_alpha_c__d1_d", "__c_alpha_c_d2_d_2_"],
        ),
    ],
)
def test_categorical_descriptor_to_descriptor_encoding(
    key,
    categories,
    samples_in,
    descriptors,
):
    c = CategoricalInput(
        key=key,
        categories=categories,
        descriptors=Descriptors(
            columns={descriptors[0]: [1, 3, 5], descriptors[1]: [2, 4, 6]}
        ),
    )
    samples = pd.Series(samples_in)
    t_samples = DescriptorEncoding().encode(c, samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[3.0, 4.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0]],
            columns=[f"{key}_{des_str}" for des_str in descriptors],
        ),
    )
    untransformed = DescriptorEncoding().decode(c, t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize(
    "key, categories, descriptors",
    [
        ("c", ["B", "A", "C"], ["d1", "d2"]),
        ("c_alpha", ["B_b", "_A_a", "C_c_"], ["_d1_d", "d2_d_2_"]),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
            ["__c_alpha_c__d1_d", "__c_alpha_c_d2_d_2_"],
        ),
    ],
)
def test_categorical_descriptor_from_descriptor_encoding(key, categories, descriptors):
    c1 = CategoricalInput(
        key=key,
        categories=categories,
        descriptors=Descriptors(
            columns={descriptors[0]: [1, 3, 5], descriptors[1]: [2, 4, 6]}
        ),
    )
    descriptor_values = pd.DataFrame(
        columns=[f"{key}_{des_str}" for des_str in descriptors] + ["misc"],
        data=[[1.05, 2.5, 6], [4, 4.5, 9]],
    )
    samples = DescriptorEncoding().decode(c1, descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series([categories[0], categories[1]]))

    c2 = CategoricalInput(
        key=key,
        categories=categories,
        descriptors=Descriptors(
            columns={descriptors[0]: [1, 3, 5], descriptors[1]: [2, 4, 6]}
        ),
        allowed=[False, True, True],
    )

    samples = DescriptorEncoding().decode(c2, descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series([categories[1], categories[1]]))


@pytest.mark.parametrize(
    "input_feature, expected_with_values, expected",
    [
        (
            CategoricalInput(
                key="if1",
                categories=["a", "b"],
                allowed=[True, True],
                descriptors=Descriptors(columns={"alpha": [1, 3], "beta": [2, 4]}),
            ),
            ([1, 2], [3, 4]),
            ([1, 2], [3, 4]),
        ),
        (
            CategoricalInput(
                key="if2",
                categories=["a", "b", "c"],
                allowed=[True, False, True],
                descriptors=Descriptors(
                    columns={"alpha": [1, 3, 1], "beta": [2, 4, 5]}
                ),
            ),
            ([1, 2], [3, 5]),
            ([1, 2], [1, 5]),
        ),
        # (CategoricalInput(key="if2", categories = ["a","b"], allowed = [True, True]), ["a","b"]),
        # (CategoricalInput(key="if3", categories = ["a","b"], allowed = [True, False]), ["a"]),
        # (CategoricalInput(key="if4", categories = ["a","b"], allowed = [True, False]), ["a", "b"]),
        # (ContinuousInput(key="if1", lower_bound=2.5, upper_bound=2.9), (1,3.)),
        # (ContinuousInput(key="if2", lower_bound=1., upper_bound=3.), (1,3.)),
        # (ContinuousInput(key="if2", lower_bound=1., upper_bound=1.), (1,1.)),
    ],
)
def test_categorical_descriptor_feature_get_bounds(
    input_feature,
    expected_with_values,
    expected,
):
    experiments = pd.DataFrame(
        {"if1": ["a", "b"], "if2": ["a", "c"], "if3": ["a", "a"], "if4": ["b", "b"]},
    )
    lower, upper = DescriptorEncoding().get_bounds(
        input_feature,
        values=experiments[input_feature.key],
    )
    assert np.allclose(lower, expected_with_values[0])
    assert np.allclose(upper, expected_with_values[1])
    lower, upper = DescriptorEncoding().get_bounds(
        input_feature,
        values=None,
    )
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


def test_descriptor_encoding_filter_prunes_correlated_columns():
    """`filter_descriptors` drops correlated columns across the assembled block."""
    from bofire.data_models.features.api import CategoricalInput

    feat = CategoricalInput(
        key="c",
        categories=["a", "b", "c"],
        descriptors=Descriptors(
            columns={
                "d1": [1.0, 2.0, 3.0],
                "d2": [2.0, 4.0, 6.0],
                "d3": [0.0, 1.0, 0.0],
            }
        ),
    )
    # without filtering, all three columns are used
    assert DescriptorEncoding().get_names(feat) == ["c_d1", "c_d2", "c_d3"]
    # with filtering, the collinear d2 is dropped (the earlier d1 is kept)
    filtered = DescriptorEncoding(filter_descriptors=True).get_names(feat)
    assert filtered == ["c_d1", "c_d3"]


# --- from generators run on a structure column ---


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "key, transform_type, values",
    [
        (
            "molecule_2_two",
            Fingerprints(n_bits=32),
            {
                "molecule_2_two_fingerprint_0": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_1": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                "molecule_2_two_fingerprint_2": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "molecule_2_two_fingerprint_3": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_4": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_5": {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_6": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "molecule_2_two_fingerprint_7": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                "molecule_2_two_fingerprint_8": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_9": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_10": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_11": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_12": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_13": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_14": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_15": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_16": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0},
                "molecule_2_two_fingerprint_17": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_18": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_19": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_20": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_21": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_22": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_23": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_24": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_25": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_26": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_27": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_28": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_2_two_fingerprint_29": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_2_two_fingerprint_30": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "molecule_2_two_fingerprint_31": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            },
        ),
        (
            "molecule_",
            Fragments(
                fragments=["fr_unbrch_alkane", "fr_thiocyan"],
            ),
            {
                "molecule__fr_unbrch_alkane": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule__fr_thiocyan": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            },
        ),
        (
            "_mo_le_cule",
            MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            {
                "_mo_le_cule_NssCH2": {
                    0: 0.5963718820861676,
                    1: -1.5,
                    2: -0.28395061728395066,
                    3: -8.34319526627219,
                },
                "_mo_le_cule_ATSC2d": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
            },
        ),
    ],
)
def test_categorical_with_structure_to_descriptor_encoding(key, transform_type, values):
    input_feature = CategoricalInput(
        key=key,
        categories=VALID_SMILES.tolist(),
        descriptors=Descriptors(structure=list(VALID_SMILES.tolist())),
    )

    encoded = DescriptorEncoding(columns=[], generators=[transform_type]).encode(
        input_feature, VALID_SMILES
    )
    assert len(encoded.columns) == len(transform_type.get_descriptor_names())
    assert len(encoded) == len(smiles)
    assert_frame_equal(encoded, pd.DataFrame.from_dict(values))


@pytest.mark.parametrize(
    "key",
    [
        ("molecule_2_two"),
        ("molecule_"),
        ("molecule"),
        ("_mo_le_cule"),
    ],
)
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_with_structure_from_descriptor_encoding(key):
    feat = CategoricalInput(
        key=key,
        categories=VALID_SMILES.to_list(),
        descriptors=Descriptors(structure=list(VALID_SMILES.to_list())),
    )
    values = pd.Series(data=["c1ccccc1", "[CH3][CH2][OH]"], name=key)
    for transform_type in [
        Fingerprints(),
        Fingerprints(),
        Fragments(),
        Fragments(),
        MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
    ]:
        encoding = DescriptorEncoding(columns=[], generators=[transform_type])
        encoded = encoding.encode(feat, values=values)
        decoded = encoding.decode(feat, values=encoded)
        assert np.all(decoded == values)
