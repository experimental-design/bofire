import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bofire.data_models.enum import CategoricalEncodingEnum
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
    from bofire.data_models.molfeatures import names

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
def test_categorical_molecular_input_to_descriptor_encoding(
    key, transform_type, values
):
    input_feature = CategoricalMolecularInput(key=key, categories=VALID_SMILES.tolist())

    encoded = input_feature.to_descriptor_encoding(transform_type, VALID_SMILES)
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


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_molecular_input_select_mordred_descriptors():
    """Test the select_mordred_descriptors function."""
    # Create MordredDescriptors with all available descriptors
    transform_type = MordredDescriptors(descriptors=names.mordred)
    initial_descriptor_count = len(transform_type.descriptors)

    # Create CategoricalMolecularInput with the specified categories
    categories = [
        "CC(C)(C)C(=O)Nc1nc(NC(=O)C(C)(C)C)c2cc(Br)cnc2n1",
        "Cc1nn(C)c(C)c1Br",
        "COc1cccc(Br)n1",
        "CC(C)c1cc(C(C)C)c(Br)c(C(C)C)c1",
        "Cc1ccc(Br)cn1",
        "Cc1ncc(Br)cn1",
        "Brc1csc(Oc2ccccc2)n1",
        "Brc1cnc(N2CCCCC2)nc1",
        "CCc1cccc(CC)c1Br",
        "Brc1ccc2ncccc2c1",
        "Brc1ccnc2ccccc12",
        "O=[N+]([O-])c1cccc(Br)c1",
        "Cn1ncc2c(Br)cccc21",
        "COc1ccccc1Br",
        "Brc1cscn1",
        "COc1cc(Br)cc(OC)c1OC",
        "Brc1cncnc1",
        "CCn1c2ccccc2c2cc(Br)ccc21",
        "CSc1cccc(Br)c1",
        "Brc1cccc2ccccc12",
        "Brc1ccccn1",
        "Brc1cnc2ccccc2c1",
        "Brc1cccs1",
        "FC(F)(F)c1ccccc1Br",
        "Brc1cnc2ccccn12",
        "Cn1cnc2ccc(Br)cc2c1=O",
        "Cc1cscc1Br",
        "Cn1cnc(Br)c1",
        "CC(C)[Si](Oc1cccc(Br)c1)(C(C)C)C(C)C",
        "FC(F)(F)c1cc(Br)cc(C(F)(F)F)c1",
        "FC(F)(F)c1ccc(Br)cc1",
        "Brc1ccc2ccccc2c1",
        "Brc1cnn(C(c2ccccc2)(c2ccccc2)c2ccccc2)c1",
        "COc1cccc(OC)c1Br",
        "Cn1cc(Br)ccc1=O",
        "Brc1ccccc1-n1cccn1",
        "COc1ccc(Br)c(C)c1",
    ]

    feat = CategoricalMolecularInput(key="dummy", categories=categories)

    # Run select_mordred_descriptors
    feat.select_mordred_descriptors(transform_type=transform_type, cutoff=0.95)

    # Check that the number of descriptors has decreased
    final_descriptor_count = len(transform_type.descriptors)

    assert final_descriptor_count < initial_descriptor_count, (
        f"Expected descriptor count to decrease, but got "
        f"initial: {initial_descriptor_count}, final: {final_descriptor_count}"
    )

    # Verify that all remaining descriptors are valid
    assert all(
        desc in names.mordred for desc in transform_type.descriptors
    ), "Some descriptors in the filtered list are not valid Mordred descriptors"

    # Verify that the descriptor count is positive
    assert (
        final_descriptor_count > 0
    ), "All descriptors were removed, expected at least some to remain"
