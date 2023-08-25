import importlib

import pandas as pd
import pytest

import bofire.data_models.molfeatures.names as names
from bofire.data_models.molfeatures.api import Fragments

RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_framents():
    from rdkit.Chem import Descriptors

    assert names.fragments == [
        rdkit_fragment[0] for rdkit_fragment in Descriptors.descList[124:]
    ]


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_mordred():
    from mordred import Calculator
    from mordred import descriptors as mordred_descriptors

    calc = Calculator(mordred_descriptors, ignore_3D=False)
    assert names.mordred == [str(d) for d in calc.descriptors]


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize("fragments", [None, ["fr_Al_COO", "fr_Al_OH"]])
def test_fragments_get_contained_fragments(fragments):
    mf = Fragments(fragments=fragments)
    smiles = pd.Series(
        [
            "CC(=O)Oc1ccccc1C(=O)O",
            "c1ccccc1",
            "[CH3][CH2][OH]",
            "N[C@](C)(F)C(=O)O",
        ]
    )
    mf.get_contained_fragments(smiles)
