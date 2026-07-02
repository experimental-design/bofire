import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bofire.data_models.encodings.api import MolecularEncoding, OneHotEncoding
from bofire.data_models.features.molecular import (
    CategoricalMolecularInput,
    ContinuousMolecularInput,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

if RDKIT_AVAILABLE:
    pass

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
@pytest.mark.parametrize(
    "key, transform_type, values",
    [
        (
            "molecule_2_two",
            Fingerprints(n_bits=32, filter_descriptors=False),
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
                filter_descriptors=False,
            ),
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
                filter_descriptors=False,
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
            MordredDescriptors(
                descriptors=["NssCH2", "ATSC2d"], filter_descriptors=False
            ),
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
def test_categorical_molecular_input_to_descriptor_encoding(
    key, transform_type, values
):
    input_feature = CategoricalMolecularInput(key=key, categories=VALID_SMILES.tolist())

    encoded = MolecularEncoding(generator=transform_type).encode(
        input_feature, VALID_SMILES
    )
    assert len(encoded.columns) == len(transform_type.get_descriptor_names())
    assert len(encoded) == len(smiles)
    assert_frame_equal(encoded, pd.DataFrame.from_dict(values))


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_molecular_input_invalid_smiles():
    with pytest.raises(ValueError, match="abcd is not a valid smiles string."):
        CategoricalMolecularInput(
            key="a",
            categories=["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "abcd"],
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_continous_molecular_input_valid_smiles():
    with pytest.raises(ValueError, match="abc is not a valid smiles string"):
        ContinuousMolecularInput(key="a", bounds=[0, 1], molecule="abc")


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
        encoding = MolecularEncoding(generator=transform_type)
        encoded = encoding.encode(feat, values=values)
        decoded = encoding.decode(feat, values=encoded)
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
        transform_type=OneHotEncoding(),
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
    lower, upper = MolecularEncoding(
        generator=MordredDescriptors(
            descriptors=[
                "nAromAtom",
                "nAromBond",
            ],
            filter_descriptors=False,
        ),
    ).get_bounds(feat)
    assert lower == [6.0, 6.0]
    assert upper == [6.0, 6.0]

    lower, upper = MolecularEncoding(
        generator=MordredDescriptors(
            descriptors=[
                "nAromAtom",
                "nAromBond",
            ],
            filter_descriptors=False,
        ),
    ).get_bounds(feat, values=VALID_SMILES)
    assert lower == [0.0, 0.0]
    assert upper == [6.0, 6.0]


def test_categorical_molecular_input_to_pydantic_field():
    from typing import Literal

    feat = CategoricalMolecularInput(key="mol", categories=["CCO", "CC"])
    field_type, field_info = feat.to_pydantic_field()
    assert field_type == Literal["CCO", "CC"]
    assert (
        field_info.description
        == "Categorical, allowed: ['CCO', 'CC'] — descriptors: ['smiles']"
    )


def test_categorical_molecular_input_to_pydantic_field_falls_back_above_threshold():
    from bofire.data_models.features.categorical import LLM_ENUM_SCHEMA_THRESHOLD

    # Generate enough distinct SMILES by varying alkane chain length
    smiles = ["C" * (i + 1) for i in range(LLM_ENUM_SCHEMA_THRESHOLD + 1)]
    feat = CategoricalMolecularInput(key="mol", categories=smiles)
    field_type, field_info = feat.to_pydantic_field()
    assert field_type is str
    # description still lists the SMILES so the LLM has guidance
    assert smiles[0] in field_info.description
    assert smiles[-1] in field_info.description


def test_continuous_molecular_input_to_pydantic_field():
    feat = ContinuousMolecularInput(key="conc", molecule="CCO", bounds=(0.0, 1.0))
    _, field_info = feat.to_pydantic_field()
    assert (
        field_info.description
        == "Continuous molecular (SMILES: CCO), bounds [0.0, 1.0]"
    )
