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
from bofire.data_models.encodings.api import DescriptorEncoding, OneHotEncoding
from bofire.data_models.features._descriptors import Descriptors
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
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
def test_categorical_molecular_input_to_descriptor_encoding(
    key, transform_type, values
):
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


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_molecular_input_invalid_smiles():
    with pytest.raises(ValueError, match="abcd is not a valid smiles string."):
        CategoricalInput(
            key="a",
            categories=["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "abcd"],
            descriptors=Descriptors(
                structure=["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "abcd"]
            ),
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_continous_molecular_input_valid_smiles():
    with pytest.raises(ValueError, match="abc is not a valid smiles string"):
        ContinuousInput(
            key="a", bounds=[0, 1], descriptors=Descriptors(structure=["abc"])
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_molecular_input_valid_smiles():
    CategoricalInput(
        key="a",
        categories=VALID_SMILES.tolist(),
        descriptors=Descriptors(structure=list(VALID_SMILES.tolist())),
    )


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


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_molecular_input_get_bounds():
    # first test with onehot
    feat = CategoricalInput(
        key="a",
        categories=VALID_SMILES.to_list(),
        allowed=[True, True, True, True],
        descriptors=Descriptors(structure=list(VALID_SMILES.to_list())),
    )
    lower, upper = feat.get_bounds(
        transform_type=OneHotEncoding(),
        reference_value=None,
    )
    assert lower == [0 for _ in range(len(feat.categories))]
    assert upper == [1 for _ in range(len(feat.categories))]
    # now test it with descriptors,
    feat = CategoricalInput(
        key="a",
        categories=VALID_SMILES.to_list(),
        allowed=[True, True, False, False],
        descriptors=Descriptors(structure=list(VALID_SMILES.to_list())),
    )
    lower, upper = DescriptorEncoding(
        columns=[],
        generators=[
            MordredDescriptors(
                descriptors=[
                    "nAromAtom",
                    "nAromBond",
                ],
            ),
        ],
    ).get_bounds(feat)
    assert lower == [6.0, 6.0]
    assert upper == [6.0, 6.0]

    lower, upper = DescriptorEncoding(
        columns=[],
        generators=[
            MordredDescriptors(
                descriptors=[
                    "nAromAtom",
                    "nAromBond",
                ],
            ),
        ],
    ).get_bounds(feat, values=VALID_SMILES)
    assert lower == [0.0, 0.0]
    assert upper == [6.0, 6.0]


def test_categorical_molecular_input_to_pydantic_field():
    from typing import Literal

    feat = CategoricalInput(
        key="mol",
        categories=["CCO", "CC"],
        descriptors=Descriptors(structure=["CCO", "CC"]),
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type == Literal["CCO", "CC"]
    assert field_info.description == (
        "Categorical molecular (SMILES), allowed: ['CCO', 'CC'] — "
        "structure: ['CCO', 'CC']"
    )


def test_categorical_molecular_input_to_pydantic_field_structure_beside_names():
    """Categories need not be the SMILES themselves.

    Main could not express this -- `CategoricalMolecularInput` used the categories as the
    structures -- so without the explicit `structure` part the SMILES would be invisible
    to the model here.
    """
    feat = CategoricalInput(
        key="solvent",
        categories=["water", "ethanol"],
        descriptors=Descriptors(columns={"logP": [-1.4, -0.3]}, structure=["O", "CCO"]),
    )
    _, field_info = feat.to_pydantic_field()
    assert field_info.description == (
        "Categorical molecular (SMILES), allowed: ['water', 'ethanol'] — "
        "descriptors per category: {'water': {'logP': -1.4}, 'ethanol': {'logP': -0.3}} — "
        "structure: ['O', 'CCO']"
    )


def test_categorical_molecular_input_to_pydantic_field_falls_back_above_threshold():
    from bofire.data_models.features.categorical import LLM_ENUM_SCHEMA_THRESHOLD

    # Generate enough distinct SMILES by varying alkane chain length
    smiles = ["C" * (i + 1) for i in range(LLM_ENUM_SCHEMA_THRESHOLD + 1)]
    feat = CategoricalInput(
        key="mol", categories=smiles, descriptors=Descriptors(structure=list(smiles))
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type is str
    # description still lists the SMILES so the LLM has guidance
    assert smiles[0] in field_info.description
    assert smiles[-1] in field_info.description


def test_continuous_molecular_input_to_pydantic_field():
    feat = ContinuousInput(
        key="conc", bounds=(0.0, 1.0), descriptors=Descriptors(structure=["CCO"])
    )
    _, field_info = feat.to_pydantic_field()
    # a numeric feature is a single component, so it carries exactly one structure
    assert (
        field_info.description
        == "Continuous molecular (SMILES: CCO), bounds [0.0, 1.0]"
    )


def test_continuous_molecular_input_to_pydantic_field_with_descriptors():
    feat = ContinuousInput(
        key="conc",
        bounds=(0.0, 1.0),
        descriptors=Descriptors(columns={"logP": [-0.3]}, structure=["CCO"]),
    )
    _, field_info = feat.to_pydantic_field()
    assert field_info.description == (
        "Continuous molecular (SMILES: CCO), bounds [0.0, 1.0] — "
        "descriptors: {'logP': -0.3}"
    )


def test_discrete_input_to_pydantic_field_with_descriptors():
    """`DiscreteInput` gained descriptors in this refactor; main had none.

    A restricted amount of a substance still describes one substance, so the block has a
    single level here just as it does on `ContinuousInput`.
    """
    feat = DiscreteInput(
        key="loading",
        values=[1.0, 2.0, 5.0],
        descriptors=Descriptors(columns={"logP": [-0.3]}, structure=["CCO"]),
    )
    _, field_info = feat.to_pydantic_field()
    assert field_info.description == (
        "Discrete molecular (SMILES: CCO), allowed values: [1.0, 2.0, 5.0] — "
        "descriptors: {'logP': -0.3}"
    )
