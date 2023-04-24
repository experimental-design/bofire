import importlib

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from bofire.data_models.features.molecular import MolecularInput

RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

VALID_SMILES = pd.Series(
    [
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccccc1",
        "[CH3][CH2][OH]",
        "C-C-O",
        "OCC",
        "N[C@](C)(F)C(=O)O",
    ]
)
INVALID_SMILES = pd.Series(["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "abcd"])


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molecular_input_validate_experimental():
    m = MolecularInput(key="molecules")
    vals = m.validate_experimental(VALID_SMILES)
    assert_series_equal(vals, VALID_SMILES)
    with pytest.raises(ValueError):
        m.validate_experimental(INVALID_SMILES)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molecular_input_validate_candidental():
    m = MolecularInput(key="molecules")
    vals = m.validate_candidental(VALID_SMILES)
    assert_series_equal(vals, VALID_SMILES)
    with pytest.raises(ValueError):
        m.validate_candidental(INVALID_SMILES)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_molecular_input_fixed():
    m = MolecularInput(key="molecules")
    assert m.fixed_value() is None
    assert m.is_fixed() is False


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_to_fingerprints():
    m = MolecularInput(key="molecules")
    data = m.to_fingerprints(VALID_SMILES)
    assert data.shape[0] == 6
    with pytest.raises(ValueError):
        m.to_fingerprints(INVALID_SMILES)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_to_bag_of_characters():
    m = MolecularInput(key="molecules")
    data = m.to_bag_of_characters(VALID_SMILES)
    assert data.shape[0] == 6
    with pytest.raises(ValueError):
        m.to_bag_of_characters(INVALID_SMILES)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_to_fragments():
    m = MolecularInput(key="molecules")
    data = m.to_fragments(VALID_SMILES)
    assert data.shape[0] == 6
    with pytest.raises(ValueError):
        m.to_fragments(INVALID_SMILES)
