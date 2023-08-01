import importlib

import pytest

import bofire.data_models.molfeatures.names as names

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
