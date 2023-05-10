import importlib

import pytest

from bofire.utils.cheminformatics import (
    smiles2bag_of_characters,
    smiles2fingerprints,
    smiles2fragments,
    smiles2mol,
)

RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2mol():
    # invalid
    with pytest.raises(ValueError):
        smiles2mol("abcd")
    # valid
    smiles2mol("CC(=O)Oc1ccccc1C(=O)O")


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2bag_of_characters():
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    desc = smiles2bag_of_characters(smiles=smiles)
    assert desc.shape[0] == 2


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2fingerprints():
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    desc = smiles2fingerprints(smiles=smiles, n_bits=512)
    assert desc.shape[0] == 2
    assert desc.shape[1] == 512


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_smiles2fragments():
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"]
    desc = smiles2fragments(smiles=smiles)
    assert desc.shape[0] == 2
