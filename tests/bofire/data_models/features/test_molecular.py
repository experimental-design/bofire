import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.molecular import (
    CategoricalMolecularInput,
    MolecularInput,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "[CH3][CH2][OH]",
    "N[C@](C)(F)C(=O)O",
]
VALID_SMILES = pd.Series(smiles)
VALID_SMILES.name = "molecule"
INVALID_SMILES = pd.Series(["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "abcd"])


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molecular_input_validate_experimental():
    m = MolecularInput(key="molecule")
    vals = m.validate_experimental(VALID_SMILES)
    assert_series_equal(vals, VALID_SMILES)
    with pytest.raises(ValueError):
        m.validate_experimental(INVALID_SMILES)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molecular_input_validate_candidental():
    m = MolecularInput(key="molecule")
    vals = m.validate_candidental(VALID_SMILES)
    assert_series_equal(vals, VALID_SMILES)
    with pytest.raises(ValueError):
        m.validate_candidental(INVALID_SMILES)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molecular_input_fixed():
    m = MolecularInput(key="molecule")
    assert m.fixed_value() is None
    assert m.is_fixed() is False


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "expected, transform_type",
    [
        (
            (
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                ],
            ),
            Fingerprints(n_bits=32),
        ),
        (
            (
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 1.0],
            ),
            Fragments(fragments=["fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO"]),
        ),
        (
            (
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                ],
            ),
            FingerprintsFragments(
                n_bits=32,
                fragments=["fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO"],
            ),
        ),
        (
            ([-8.34319526627219, 0.0], [0.5963718820861676, 1.0]),
            MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
        ),
    ],
)
def test_molecular_feature_get_bounds(expected, transform_type):
    input_feature = MolecularInput(key="molecule")
    lower, upper = input_feature.get_bounds(
        transform_type=transform_type,
        values=VALID_SMILES,
        reference_value=None,
    )
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


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
            Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            {
                "molecule__fr_unbrch_alkane": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule__fr_thiocyan": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            },
        ),
        (
            "molecule",
            FingerprintsFragments(
                n_bits=32,
                fragments=["fr_unbrch_alkane", "fr_thiocyan"],
            ),
            {
                "molecule_fingerprint_0": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_1": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                "molecule_fingerprint_2": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "molecule_fingerprint_3": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_4": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_5": {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_6": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "molecule_fingerprint_7": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
                "molecule_fingerprint_8": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_9": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_10": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_11": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_12": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_13": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_14": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_15": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_16": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0},
                "molecule_fingerprint_17": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_18": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_19": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_20": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_21": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_22": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_23": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_24": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_25": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_26": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_27": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_28": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fingerprint_29": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "molecule_fingerprint_30": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "molecule_fingerprint_31": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fr_unbrch_alkane": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fr_thiocyan": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
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
def test_molecular_input_to_descriptor_encoding(key, transform_type, values):
    input_feature = MolecularInput(key=key)

    encoded = input_feature.to_descriptor_encoding(transform_type, VALID_SMILES)
    assert len(encoded.columns) == len(transform_type.get_descriptor_names())
    assert len(encoded) == len(smiles)
    assert_frame_equal(encoded, pd.DataFrame.from_dict(values))


### tests for CategoricalMolecularInput
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_molecular_input_invalid_smiles():
    with pytest.raises(ValueError, match="abcd is not a valid smiles string."):
        CategoricalMolecularInput(
            key="a",
            categories=["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "abcd"],
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_molecular_input_valid_smiles():
    CategoricalMolecularInput(key="a", categories=VALID_SMILES.tolist())


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
def test_categorical_molecular_input_from_descriptor_encoding(key):
    feat = CategoricalMolecularInput(key=key, categories=VALID_SMILES.to_list())
    values = pd.Series(data=["c1ccccc1", "[CH3][CH2][OH]"], name=key)
    for transform_type in [
        Fingerprints(),
        FingerprintsFragments(),
        Fragments(),
        MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
    ]:
        encoded = feat.to_descriptor_encoding(transform_type, values=values)
        decoded = feat.from_descriptor_encoding(transform_type, values=encoded)
        assert np.all(decoded == values)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_molecular_input_get_bounds():
    # first test with onehot
    feat = CategoricalMolecularInput(
        key="a",
        categories=VALID_SMILES.to_list(),
        allowed=[True, True, True, True],
    )
    lower, upper = feat.get_bounds(
        transform_type=CategoricalEncodingEnum.ONE_HOT,
        reference_value=None,
    )
    assert lower == [0 for _ in range(len(feat.categories))]
    assert upper == [1 for _ in range(len(feat.categories))]
    # now test it with descriptors,
    feat = CategoricalMolecularInput(
        key="a",
        categories=VALID_SMILES.to_list(),
        allowed=[True, True, False, False],
    )
    lower, upper = feat.get_bounds(
        transform_type=MordredDescriptors(
            descriptors=[
                "nAromAtom",
                "nAromBond",
            ],
        ),
        reference_value=None,
    )
    assert lower == [6.0, 6.0]
    assert upper == [6.0, 6.0]

    lower, upper = feat.get_bounds(
        transform_type=MordredDescriptors(
            descriptors=[
                "nAromAtom",
                "nAromBond",
            ],
        ),
        values=VALID_SMILES,
        reference_value=None,
    )
    assert lower == [0.0, 0.0]
    assert upper == [6.0, 6.0]
