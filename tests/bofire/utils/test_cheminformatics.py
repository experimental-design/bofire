import importlib

import numpy as np
import pytest

from bofire.utils.cheminformatics import (  # smiles2bag_of_characters,
    smiles2fingerprints,
    smiles2fragments,
    smiles2fragments_fingerprints,
    smiles2mol,
    smiles2mordred,
)


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2mol():
    # invalid
    with pytest.raises(ValueError):
        smiles2mol("abcd")
    # valid
    smiles2mol("CC(=O)Oc1ccccc1C(=O)O")


smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "[CH3][CH2][OH]",
    "N[C@](C)(F)C(=O)O",
]


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2fingerprints():
    values = np.array(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                1,
                1,
                0,
                0,
            ],
            [
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
            ],
        ],
    )
    desc = smiles2fingerprints(smiles=smiles, n_bits=32)
    assert desc.shape[0] == 4
    assert desc.shape[1] == 32
    np.testing.assert_array_equal(desc, values)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2fragments():
    values = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    desc = smiles2fragments(
        smiles=smiles,
        fragments_list=["fr_unbrch_alkane", "fr_thiocyan"],
    )
    assert desc.shape[0] == 4
    assert desc.shape[1] == 2
    np.testing.assert_array_equal(desc, values)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2fragments_fingerprints():
    values = np.array(
        [
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
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
                0.0,
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
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
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
                1.0,
                1.0,
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
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
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
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ],
    )

    desc = smiles2fragments_fingerprints(
        smiles=smiles,
        n_bits=32,
        fragments_list=["fr_unbrch_alkane", "fr_thiocyan"],
    )
    assert desc.shape[0] == 4
    assert desc.shape[1] == 32 + 2
    np.testing.assert_array_equal(desc, values)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2mordred():
    values = np.array(
        [
            [0.5963718820861676, 0.0],
            [-1.5, 0.0],
            [-0.28395061728395066, 1.0],
            [-8.34319526627219, 0.0],
        ],
    )

    desc = smiles2mordred(smiles=smiles, descriptors_list=["NssCH2", "ATSC2d"])
    assert desc.shape[0] == 4
    assert desc.shape[1] == 2
    np.testing.assert_array_almost_equal(desc, values)


# @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
# def test_smiles2bag_of_characters():
#     smiles = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
#     desc = smiles2bag_of_characters(smiles=smiles)
#     assert desc.shape[0] == 2
