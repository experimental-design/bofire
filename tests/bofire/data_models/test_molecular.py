import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from bofire.data_models.features.molecular import MolecularInput
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)

try:
    from rdkit.Chem import Descriptors
except ImportError:
    warnings.warn(
        "rdkit not installed, BoFire's cheminformatics utilities cannot be used."
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
    "molfeatures, expected",
    [
        (Fingerprints(), [f"fingerprint_{i}" for i in range(2048)]),
        (Fingerprints(n_bits=32), [f"fingerprint_{i}" for i in range(32)]),
        (
            Fragments(),
            [rdkit_fragment[0] for rdkit_fragment in Descriptors.descList[124:]],
        ),
        (
            Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            ["fr_unbrch_alkane", "fr_thiocyan"],
        ),
        (
            FingerprintsFragments(),
            [f"fingerprint_{i}" for i in range(2048)]
            + [rdkit_fragment[0] for rdkit_fragment in Descriptors.descList[124:]],
        ),
        (
            FingerprintsFragments(
                n_bits=32, fragments=["fr_unbrch_alkane", "fr_thiocyan"]
            ),
            [f"fingerprint_{i}" for i in range(32)]
            + ["fr_unbrch_alkane", "fr_thiocyan"],
        ),
        (MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]), ["NssCH2", "ATSC2d"]),
    ],
)
def test_molfeatures_get_descriptor_names(molfeatures, expected):
    assert molfeatures.get_descriptor_names() == expected


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
                n_bits=32, fragments=["fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO"]
            ),
        ),
        (
            ([-8.34319526627219, 0.0], [0.5963718820861676, 1.0]),
            MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
        ),
    ],
)
def test_molecular_descriptor_feature_get_bounds(expected, transform_type):
    input_feature = MolecularInput(key="molecule")
    lower, upper = input_feature.get_bounds(
        transform_type=transform_type,
        values=VALID_SMILES,
    )
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "transform_type, values",
    [
        (
            Fingerprints(n_bits=32),
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
            },
        ),
        (
            Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            {
                "molecule_fr_unbrch_alkane": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                "molecule_fr_thiocyan": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            },
        ),
        (
            FingerprintsFragments(
                n_bits=32, fragments=["fr_unbrch_alkane", "fr_thiocyan"]
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
            MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            {
                "molecule_NssCH2": {
                    0: 0.5963718820861676,
                    1: -1.5,
                    2: -0.28395061728395066,
                    3: -8.34319526627219,
                },
                "molecule_ATSC2d": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
            },
        ),
    ],
)
def test_molecular_input_to_descriptor_encoding(transform_type, values):
    input_feature = MolecularInput(key="molecule")

    encoded = input_feature.to_descriptor_encoding(transform_type, VALID_SMILES)
    assert len(encoded.columns) == len(transform_type.get_descriptor_names())
    assert len(encoded) == len(smiles)
    assert_frame_equal(encoded, pd.DataFrame.from_dict(values))


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molfeatures_type_get_descriptor_values_fingerprints():
    values = {
        "fingerprint_0": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
        "fingerprint_1": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
        "fingerprint_2": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
        "fingerprint_3": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_4": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_5": {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0},
        "fingerprint_6": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
        "fingerprint_7": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
        "fingerprint_8": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_9": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_10": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_11": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_12": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_13": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_14": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_15": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_16": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0},
        "fingerprint_17": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
        "fingerprint_18": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_19": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_20": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_21": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_22": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_23": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_24": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_25": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_26": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_27": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_28": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_29": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_30": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
        "fingerprint_31": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
    }

    molfeature = Fingerprints(n_bits=32)
    generated = molfeature.get_descriptor_values(VALID_SMILES)
    assert_frame_equal(generated, pd.DataFrame.from_dict(values))


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molfeatures_type_get_descriptor_values_fragments():
    values = {
        "fr_unbrch_alkane": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fr_thiocyan": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
    }

    molfeature = Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"])
    generated = molfeature.get_descriptor_values(VALID_SMILES)
    assert_frame_equal(generated, pd.DataFrame.from_dict(values))


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "fragment_list",
    [
        (["fr_unbrch_alkane','fr_unbrch_alkane', 'fr_thiocyan"]),
        (["frag','fr_unbrch_alkane', 'fr_thiocyan"]),
    ],
)
def test_molfeatures_type_fragments_invalid(fragment_list):
    with pytest.raises(ValueError):
        FingerprintsFragments(fragments=fragment_list)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molfeatures_type_get_descriptor_values_fingerprintsfragments():
    values = {
        "fingerprint_0": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
        "fingerprint_1": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
        "fingerprint_2": {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0},
        "fingerprint_3": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_4": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_5": {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0},
        "fingerprint_6": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
        "fingerprint_7": {0: 1.0, 1: 0.0, 2: 1.0, 3: 1.0},
        "fingerprint_8": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_9": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_10": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_11": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_12": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_13": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_14": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_15": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_16": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0},
        "fingerprint_17": {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
        "fingerprint_18": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_19": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_20": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_21": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_22": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_23": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_24": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_25": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_26": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_27": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_28": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fingerprint_29": {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0},
        "fingerprint_30": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
        "fingerprint_31": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fr_unbrch_alkane": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        "fr_thiocyan": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
    }

    molfeature = FingerprintsFragments(
        n_bits=32, fragments=["fr_unbrch_alkane", "fr_thiocyan"]
    )
    generated = molfeature.get_descriptor_values(VALID_SMILES)
    assert_frame_equal(generated, pd.DataFrame.from_dict(values))


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "fragment_list",
    [
        (["fr_unbrch_alkane','fr_unbrch_alkane', 'fr_thiocyan"]),
        (["frag','fr_unbrch_alkane', 'fr_thiocyan"]),
    ],
)
def test_molfeatures_type_fingerprintsfragments_invalid(fragment_list):
    with pytest.raises(ValueError):
        FingerprintsFragments(fragments=fragment_list)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molfeatures_type_get_descriptor_values_mordreddescriptors():
    values = {
        "NssCH2": {
            0: 0.5963718820861676,
            1: -1.5,
            2: -0.28395061728395066,
            3: -8.34319526627219,
        },
        "ATSC2d": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
    }

    molfeature = MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])
    generated = molfeature.get_descriptor_values(VALID_SMILES)
    assert_frame_equal(generated, pd.DataFrame.from_dict(values))


@pytest.mark.parametrize(
    "mordred_list",
    [
        (["NssCH2", "NssCH2", "ATSC2d"]),
        (["desc", "NssCH2", "ATSC2d"]),
    ],
)
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molfeatures_type_mordreddescriptors_invalid(mordred_list):
    with pytest.raises(ValueError):
        MordredDescriptors(descriptors=mordred_list)
