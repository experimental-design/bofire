import importlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bofire.data_models.molfeatures import names
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


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_fragments():
    from rdkit.Chem import Descriptors

    fragment_list = [
        item[0] for item in Descriptors.descList if item[0].startswith("fr_")
    ]

    assert names.fragments == fragment_list


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_mordred():
    from mordred import Calculator
    from mordred import descriptors as mordred_descriptors

    calc = Calculator(mordred_descriptors, ignore_3D=False)
    assert names.mordred == [str(d) for d in calc.descriptors]


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "molfeatures, expected",
    [
        (Fingerprints(), [f"fingerprint_{i}" for i in range(2048)]),
        (Fingerprints(n_bits=32), [f"fingerprint_{i}" for i in range(32)]),
        (
            Fragments(),
            names.fragments,
        ),
        (
            Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            ["fr_unbrch_alkane", "fr_thiocyan"],
        ),
        (
            FingerprintsFragments(),
            [f"fingerprint_{i}" for i in range(2048)] + names.fragments,
        ),
        (
            FingerprintsFragments(
                n_bits=32,
                fragments=["fr_unbrch_alkane", "fr_thiocyan"],
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
        n_bits=32,
        fragments=["fr_unbrch_alkane", "fr_thiocyan"],
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
